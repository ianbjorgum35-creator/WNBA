"""Turns a raw player game log into the recency windows, hit rates, minutes
profile, usage profile, and teammate on/off splits the projection model needs.

All functions operate on a normalized pandas DataFrame (see normalize_gamelog)
so they work identically whether the log came from data_sources (auto-fetch)
or a manually uploaded CSV.
"""

from typing import Optional

import numpy as np
import pandas as pd

from . import config

EXPECTED_NUMERIC_COLS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]


def normalize_gamelog(raw_rows) -> pd.DataFrame:
    """Accepts a list of dict rows (from data_sources.get_player_gamelog, or
    a manually uploaded CSV read with pandas) and returns a clean, ascending
    chronological DataFrame with numeric stat columns."""
    df = pd.DataFrame(raw_rows).copy()
    if df.empty:
        return df

    rename_map = {"WL": "WIN_LOSS", "Wl": "WIN_LOSS"}
    df = df.rename(columns=rename_map)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

    for col in EXPECTED_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "MIN" in df.columns:
        # stats.wnba.com sometimes returns "MM:SS" strings for MIN
        def _parse_min(v):
            if isinstance(v, str) and ":" in v:
                m, s = v.split(":")
                return float(m) + float(s) / 60.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        df["MIN"] = df["MIN"].apply(_parse_min)

    return df


def stat_series(df: pd.DataFrame, prop_key: str) -> pd.Series:
    """Per-game value of the requested prop (summed across its component
    columns, e.g. PRA = PTS + REB + AST)."""
    prop = config.PROP_TYPES_BY_KEY[prop_key]
    cols = [c for c in prop.columns if c in df.columns]
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].sum(axis=1, skipna=False)


def _hit_rate(values: pd.Series, line: float, direction: str) -> Optional[float]:
    values = values.dropna()
    if values.empty:
        return None
    if direction == "Over":
        decided = values[values != line]
        hits = (decided > line).sum()
    else:
        decided = values[values != line]
        hits = (decided < line).sum()
    if len(decided) == 0:
        return None
    return hits / len(decided)


def rolling_windows(df: pd.DataFrame, prop_key: str, line: float, direction: str) -> dict:
    """Season / L20 / L10 / L5 averages, std devs, and hit rates vs. `line`."""
    values = stat_series(df, prop_key)
    n = len(values)

    def _window(w):
        return values.tail(w) if n >= 1 else pd.Series(dtype=float)

    windows = {"season": values, "l20": _window(20), "l10": _window(10), "l5": _window(5)}
    out = {}
    for name, series in windows.items():
        series = series.dropna()
        out[f"{name}_avg"] = float(series.mean()) if len(series) else None
        out[f"{name}_std"] = float(series.std(ddof=1)) if len(series) > 1 else None
        out[f"{name}_games"] = int(len(series))
        if name != "season":
            out[f"{name}_hit_rate"] = _hit_rate(series, line, direction)
    out["season_hit_rate"] = _hit_rate(values, line, direction)
    return out


ROTATION_THRESHOLDS = {"starter": 28.0, "rotation": 15.0}


def minutes_profile(df: pd.DataFrame) -> dict:
    minutes = df["MIN"].dropna() if "MIN" in df.columns else pd.Series(dtype=float)
    if minutes.empty:
        return {
            "season_avg_min": None, "l10_avg_min": None, "l5_avg_min": None,
            "starter_pct": None, "rotation_role": "Unknown",
        }
    season_avg = float(minutes.mean())
    l10_avg = float(minutes.tail(10).mean())
    l5_avg = float(minutes.tail(5).mean())

    starter_pct = None
    if "START_POSITION" in df.columns:
        started = df["START_POSITION"].fillna("").astype(str).str.strip() != ""
        starter_pct = float(started.tail(10).mean())

    recent = l5_avg if not np.isnan(l5_avg) else l10_avg
    if recent >= ROTATION_THRESHOLDS["starter"]:
        role = "Starter"
    elif recent >= ROTATION_THRESHOLDS["rotation"]:
        role = "Rotation"
    else:
        role = "Bench / Deep Rotation"

    return {
        "season_avg_min": season_avg, "l10_avg_min": l10_avg, "l5_avg_min": l5_avg,
        "starter_pct": starter_pct, "rotation_role": role,
    }


def usage_profile(df: pd.DataFrame) -> dict:
    if "USG_PCT" in df.columns and df["USG_PCT"].notna().any():
        usg = pd.to_numeric(df["USG_PCT"], errors="coerce")
        return {
            "season_usage_pct": float(usg.mean()),
            "l10_usage_pct": float(usg.tail(10).mean()),
            "source": "auto",
        }
    return {"season_usage_pct": None, "l10_usage_pct": None, "source": "unavailable"}


def teammate_on_off_split(player_df: pd.DataFrame, teammate_df: pd.DataFrame, prop_key: str) -> dict:
    """Compares this player's stat/minutes on dates the teammate did vs.
    didn't play, to estimate a usage/minutes bump when that teammate sits.

    Returns None-valued fields (with a note) if there isn't enough of a
    sample in either bucket to be meaningful.
    """
    if "GAME_DATE" not in player_df.columns or "GAME_DATE" not in teammate_df.columns:
        return {"bump_available": False, "note": "missing GAME_DATE column"}

    teammate_dates = set(teammate_df["GAME_DATE"].dropna().dt.date)
    player_dates = player_df["GAME_DATE"].dropna().dt.date

    with_teammate_mask = player_dates.isin(teammate_dates)
    without_teammate_mask = ~with_teammate_mask

    values = stat_series(player_df, prop_key)
    minutes = player_df["MIN"] if "MIN" in player_df.columns else pd.Series(dtype=float)

    with_n = int(with_teammate_mask.sum())
    without_n = int(without_teammate_mask.sum())

    MIN_SAMPLE = 3
    if with_n < MIN_SAMPLE or without_n < MIN_SAMPLE:
        return {
            "bump_available": False,
            "note": f"insufficient sample (with={with_n}, without={without_n} games)",
        }

    stat_with = values[with_teammate_mask].mean()
    stat_without = values[without_teammate_mask].mean()
    min_with = minutes[with_teammate_mask].mean()
    min_without = minutes[without_teammate_mask].mean()

    return {
        "bump_available": True,
        "usage_bump_pct": (stat_without / stat_with - 1.0) if stat_with else None,
        "minutes_bump_pct": (min_without / min_with - 1.0) if min_with else None,
        "with_games": with_n,
        "without_games": without_n,
    }
