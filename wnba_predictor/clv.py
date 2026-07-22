"""Bet logging, outcome recording, and CLV/outcome-driven learning.

This is the feedback loop: every prediction the user acts on gets logged;
once the game plays out (and the user records the closing line + actual
stat), we have ground truth. Two things get tuned from that history:

  1. A probability *calibrator* (logistic regression over the raw model
     probability + its adjustment components) that corrects systematic
     over/under-confidence -- classic Platt scaling.
  2. The recency-window *blend weights* (season/L20/L10/L5), nudged toward
     whichever window's hit rate has actually tracked outcomes best.

Both require a minimum sample size and always fall back to the config
defaults / an identity calibrator when there isn't enough history yet.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .odds import implied_prob_from_american, clv_percent


def ensure_log_exists(path: str = config.BET_LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=config.BET_LOG_COLUMNS).to_csv(path, index=False)


def log_bet(record: dict, path: str = config.BET_LOG_PATH) -> str:
    """Appends a prediction to the log. `record` should supply the keys
    relevant to config.BET_LOG_COLUMNS; anything missing is left blank."""
    ensure_log_exists(path)
    bet_id = str(uuid.uuid4())[:8]
    row = {col: record.get(col, "") for col in config.BET_LOG_COLUMNS}
    row["bet_id"] = bet_id
    row["logged_at"] = datetime.now(timezone.utc).isoformat()
    df = pd.read_csv(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return bet_id


def pending_bets(path: str = config.BET_LOG_PATH) -> pd.DataFrame:
    ensure_log_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df[df["hit"].isna() | (df["hit"] == "")]


def record_outcome(bet_id: str, actual_stat_value: float, closing_odds: Optional[float] = None,
                    path: str = config.BET_LOG_PATH, notes: str = "") -> dict:
    ensure_log_exists(path)
    raw = pd.read_csv(path)
    df = raw.astype(object).where(pd.notna(raw), None)
    mask = df["bet_id"] == bet_id
    if not mask.any():
        raise ValueError(f"no logged bet with id {bet_id}")

    idx = df.index[mask][0]
    line = float(df.loc[idx, "line"])
    direction = df.loc[idx, "direction"]
    hit = (actual_stat_value > line) if direction == "Over" else (actual_stat_value < line)

    df.loc[idx, "actual_stat_value"] = actual_stat_value
    df.loc[idx, "hit"] = bool(hit)
    df.loc[idx, "outcome_logged_at"] = datetime.now(timezone.utc).isoformat()
    if notes:
        df.loc[idx, "notes"] = notes

    if closing_odds is not None:
        implied_close = implied_prob_from_american(closing_odds)
        df.loc[idx, "closing_odds"] = closing_odds
        df.loc[idx, "implied_prob_close"] = implied_close
        implied_open = df.loc[idx, "implied_prob_open"]
        if implied_open is not None:
            df.loc[idx, "clv_pct"] = clv_percent(float(implied_open), implied_close)

    df.to_csv(path, index=False)
    return df.loc[idx].to_dict()


def _resolved(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    resolved = df[df["hit"].notna() & (df["hit"] != "")]
    return resolved.copy()


def calibration_report(path: str = config.BET_LOG_PATH, n_bins: int = 5) -> dict:
    ensure_log_exists(path)
    df = _resolved(pd.read_csv(path))
    if df.empty:
        return {"n_bets": 0, "message": "no resolved bets yet"}

    df["hit"] = df["hit"].astype(bool).astype(int)
    df["predicted_prob"] = pd.to_numeric(df["predicted_prob"], errors="coerce")
    df = df.dropna(subset=["predicted_prob"])

    brier = float(np.mean((df["predicted_prob"] - df["hit"]) ** 2))
    eps = 1e-6
    p = df["predicted_prob"].clip(eps, 1 - eps)
    log_loss = float(-np.mean(df["hit"] * np.log(p) + (1 - df["hit"]) * np.log(1 - p)))

    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["predicted_prob"], bins, include_lowest=True)
    bucketed = df.groupby("bin", observed=False).agg(
        n=("hit", "size"), predicted_avg=("predicted_prob", "mean"), actual_hit_rate=("hit", "mean")
    ).reset_index()

    clv_series = pd.to_numeric(df.get("clv_pct"), errors="coerce").dropna()

    return {
        "n_bets": int(len(df)),
        "win_rate": float(df["hit"].mean()),
        "brier_score": brier,
        "log_loss": log_loss,
        "avg_clv_pct": float(clv_series.mean()) if len(clv_series) else None,
        "calibration_bins": bucketed.to_dict(orient="records"),
    }


MIN_BETS_FOR_LEARNING = 20


def fit_recalibration(path: str = config.BET_LOG_PATH,
                       min_bets: int = MIN_BETS_FOR_LEARNING,
                       weights_path: str = config.LEARNED_WEIGHTS_PATH) -> dict:
    """Fits a small logistic regression (Platt scaling) mapping the raw model
    probability to a calibrated one, using resolved bet history. Persists
    coefficients to JSON so they can be loaded without scikit-learn/pickle
    version coupling.
    """
    from sklearn.linear_model import LogisticRegression

    ensure_log_exists(path)
    df = _resolved(pd.read_csv(path))
    if len(df) < min_bets:
        return {"fitted": False, "message": f"only {len(df)} resolved bets; need >= {min_bets} to (re)calibrate"}

    df["hit"] = df["hit"].astype(bool).astype(int)
    df["raw_predicted_prob"] = pd.to_numeric(df["raw_predicted_prob"], errors="coerce")
    df = df.dropna(subset=["raw_predicted_prob"])
    eps = 1e-6
    p = df["raw_predicted_prob"].clip(eps, 1 - eps)
    logit = np.log(p / (1 - p)).values.reshape(-1, 1)
    y = df["hit"].values

    model = LogisticRegression()
    model.fit(logit, y)

    weights = {
        "fitted": True,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_bets": int(len(df)),
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    return weights


def load_calibrator(weights_path: str = config.LEARNED_WEIGHTS_PATH):
    """Returns a callable(raw_prob, feature_dict) -> calibrated_prob using the
    persisted Platt-scaling coefficients, or an identity function if the
    model hasn't been fitted yet (not enough resolved bets)."""
    if not os.path.exists(weights_path):
        return lambda raw_prob, features=None: raw_prob
    with open(weights_path) as f:
        weights = json.load(f)
    if not weights.get("fitted"):
        return lambda raw_prob, features=None: raw_prob

    coef, intercept = weights["coef"], weights["intercept"]

    def _calibrate(raw_prob, features=None):
        eps = 1e-6
        p = min(max(raw_prob, eps), 1 - eps)
        logit = np.log(p / (1 - p))
        z = coef * logit + intercept
        return float(1 / (1 + np.exp(-z)))

    return _calibrate


def update_blend_weights(path: str = config.BET_LOG_PATH,
                          current_weights: Optional[dict] = None,
                          min_bets: int = MIN_BETS_FOR_LEARNING,
                          step_size: float = 0.15) -> dict:
    """Nudges the season/L20/L10/L5 blend weights toward whichever window's
    hit rate has most consistently matched actual outcomes, using each
    window's simple classification accuracy (hit_rate >= 0.5 predicting a
    win) as its performance score. Bounded step size keeps any single batch
    of new results from swinging the weights too far.
    """
    current_weights = dict(current_weights or config.DEFAULT_BLEND_WEIGHTS)
    ensure_log_exists(path)
    df = _resolved(pd.read_csv(path))
    if len(df) < min_bets:
        return {"updated": False, "message": f"only {len(df)} resolved bets; need >= {min_bets}", "weights": current_weights}

    df["hit"] = df["hit"].astype(bool).astype(int)
    windows = ["season", "l20", "l10", "l5"]
    scores = {}
    for win in windows:
        col = f"{win}_hit_rate"
        if col not in df.columns:
            scores[win] = 0.5
            continue
        rate = pd.to_numeric(df[col], errors="coerce")
        valid = rate.notna()
        if valid.sum() < min_bets // 2:
            scores[win] = 0.5
            continue
        predicted_direction = (rate[valid] >= 0.5).astype(int)
        scores[win] = float((predicted_direction == df.loc[valid, "hit"]).mean())

    total = sum(scores.values()) or 1.0
    performance_weights = {w: scores[w] / total for w in windows}

    new_weights = {
        w: (1 - step_size) * current_weights.get(w, 0.25) + step_size * performance_weights[w]
        for w in windows
    }
    norm = sum(new_weights.values())
    new_weights = {w: v / norm for w, v in new_weights.items()}

    return {"updated": True, "n_bets": int(len(df)), "window_scores": scores, "weights": new_weights}
