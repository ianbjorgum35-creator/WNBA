"""Blends recency-weighted per-minute rates with matchup/context adjustments
into a projected mean + variance, then converts that into a predicted
probability of the prop hitting.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from scipy.stats import norm, poisson

from . import config

N_TEAMS = 13


def _rank_lean(rank: Optional[float], cap: float, favorable_at_rank_1: bool) -> float:
    """Map a 1..N_TEAMS rank onto [-cap, +cap]. If favorable_at_rank_1, rank 1
    produces +cap (best possible matchup); otherwise rank 1 produces -cap."""
    if rank is None:
        return 0.0
    frac = 2 * (rank - 1) / (N_TEAMS - 1) - 1  # -1 at rank1, +1 at rankN
    lean = -frac if favorable_at_rank_1 else frac
    return max(-cap, min(cap, lean * cap))


def _bounded(value: Optional[float], cap: float) -> float:
    if value is None:
        return 0.0
    return max(-cap, min(cap, value))


@dataclass
class ProjectionInputs:
    prop_key: str
    direction: str
    line: float

    rolling: dict                      # output of stats_engine.rolling_windows
    minutes_profile: dict              # output of stats_engine.minutes_profile
    projected_minutes: Optional[float] # manual or estimated
    usage_profile: Optional[dict] = None

    opponent_pace_rank: Optional[float] = None
    opponent_def_rating_rank: Optional[float] = None
    dvp_rank: Optional[float] = None

    game_total: Optional[float] = None
    spread: Optional[float] = None
    is_home: Optional[bool] = None
    rest_days: Optional[int] = None
    is_b2b: Optional[bool] = None
    h2h: Optional[dict] = None

    teammate_usage_bump_pct: Optional[float] = None
    teammate_minutes_bump_pct: Optional[float] = None

    manual_matchup_adjustment: float = 0.0   # user slider, e.g. -0.15..0.15
    context_lean: float = 0.0                # user slider, e.g. -0.10..0.10
    context_note: str = ""

    lineup_confirmed: bool = True

    blend_weights: dict = field(default_factory=lambda: dict(config.DEFAULT_BLEND_WEIGHTS))


@dataclass
class PredictionResult:
    raw_predicted_prob: float
    predicted_prob: float           # after calibration, if any
    projected_mean: float
    projected_std: float
    base_per_min_rate: float
    adjustments: dict
    total_adjustment: float
    confidence: str
    confidence_score: float
    breakdown: dict


def _blend_rate_and_std(inputs: ProjectionInputs) -> tuple:
    r = inputs.rolling
    w = inputs.blend_weights
    windows = ["season", "l20", "l10", "l5"]

    minutes_avg = inputs.minutes_profile.get("season_avg_min") or 0.0
    l10_min = inputs.minutes_profile.get("l10_avg_min") or minutes_avg
    l5_min = inputs.minutes_profile.get("l5_avg_min") or l10_min
    window_minutes = {"season": minutes_avg, "l20": l10_min, "l10": l10_min, "l5": l5_min}

    rate_sum, weight_sum = 0.0, 0.0
    for win in windows:
        avg = r.get(f"{win}_avg")
        mins = window_minutes.get(win)
        if avg is None or not mins:
            continue
        rate = avg / mins
        wt = w.get(win, 0.0)
        rate_sum += rate * wt
        weight_sum += wt
    blended_rate = rate_sum / weight_sum if weight_sum > 0 else 0.0

    std_base = r.get("l10_std") or r.get("l20_std") or r.get("season_std") or (blended_rate * (minutes_avg or 1))
    return blended_rate, std_base


def run_prediction(inputs: ProjectionInputs,
                    calibrator: Optional[Callable[[float, dict], float]] = None) -> PredictionResult:
    prop = config.PROP_TYPES_BY_KEY[inputs.prop_key]
    caps = config.ADJUSTMENT_CAPS

    blended_rate, std_base = _blend_rate_and_std(inputs)
    projected_minutes = inputs.projected_minutes or inputs.minutes_profile.get("l5_avg_min") \
        or inputs.minutes_profile.get("season_avg_min") or 0.0

    base_mean = blended_rate * projected_minutes

    adjustments = {
        "pace": _rank_lean(inputs.opponent_pace_rank, caps["pace"], favorable_at_rank_1=True),
        "opp_defense": _rank_lean(inputs.opponent_def_rating_rank, caps["opp_defense"], favorable_at_rank_1=False),
        "dvp": _rank_lean(inputs.dvp_rank, caps["dvp"], favorable_at_rank_1=True),
        "teammate_usage_bump": _bounded(inputs.teammate_usage_bump_pct, caps["teammate_usage_bump"]),
        "teammate_minutes_bump": _bounded(inputs.teammate_minutes_bump_pct, caps["teammate_minutes_bump"]),
        "manual_matchup": _bounded(inputs.manual_matchup_adjustment, caps["manual_matchup"]),
        "context_lean": _bounded(inputs.context_lean, caps["context_lean"]),
    }

    if inputs.game_total is not None and inputs.spread is not None:
        team_implied = inputs.game_total / 2.0 - inputs.spread / 2.0
        z = max(-1.5, min(1.5, (team_implied - 80.0) / 8.0))
        adjustments["game_environment"] = caps["game_environment"] * (z / 1.5)
    else:
        adjustments["game_environment"] = 0.0

    if inputs.is_home is not None:
        adjustments["home_away"] = caps["home_away"] if inputs.is_home else -caps["home_away"] * 0.5
    else:
        adjustments["home_away"] = 0.0

    if inputs.is_b2b:
        adjustments["rest"] = -caps["rest"]
    elif inputs.rest_days is not None and inputs.rest_days >= 2:
        adjustments["rest"] = caps["rest"] * 0.5
    else:
        adjustments["rest"] = 0.0

    total_adjustment = sum(adjustments.values())
    total_adjustment = max(-0.30, min(0.30, total_adjustment))

    projected_mean = max(0.0, base_mean * (1 + total_adjustment))

    minutes_ratio = (projected_minutes / (inputs.minutes_profile.get("l5_avg_min") or projected_minutes or 1))
    projected_std = std_base * math.sqrt(max(minutes_ratio, 0.25))
    if not inputs.lineup_confirmed:
        projected_std *= config.UNCONFIRMED_LINEUP_VARIANCE_INFLATION
    projected_std = max(projected_std, projected_mean * 0.15, 0.5)

    line = inputs.line
    if prop.distribution == "poisson":
        mu = max(projected_mean, 0.05)
        floor_line = math.floor(line)
        p_over = 1 - poisson.cdf(floor_line, mu)
    else:
        z_line = (line - projected_mean) / projected_std
        p_over = 1 - norm.cdf(z_line)

    raw_prob = p_over if inputs.direction == "Over" else 1 - p_over
    raw_prob = min(max(raw_prob, 0.01), 0.99)

    calibrated_prob = raw_prob
    if calibrator is not None:
        try:
            calibrated_prob = calibrator(raw_prob, {
                "prop_key": inputs.prop_key,
                "total_adjustment": total_adjustment,
                **adjustments,
            })
            calibrated_prob = min(max(calibrated_prob, 0.01), 0.99)
        except Exception:
            calibrated_prob = raw_prob

    games_played = inputs.rolling.get("season_games", 0) or 0
    completeness_fields = [inputs.opponent_pace_rank, inputs.opponent_def_rating_rank,
                            inputs.dvp_rank, inputs.projected_minutes]
    completeness = sum(1 for f in completeness_fields if f is not None) / len(completeness_fields)

    score = 0.4 * min(games_played / 15.0, 1.0) + 0.3 * (1.0 if inputs.lineup_confirmed else 0.0) + 0.3 * completeness
    confidence = "High" if score >= 0.75 else "Medium" if score >= 0.45 else "Low"

    breakdown = {
        "season_avg": inputs.rolling.get("season_avg"),
        "l20_avg": inputs.rolling.get("l20_avg"),
        "l10_avg": inputs.rolling.get("l10_avg"),
        "l5_avg": inputs.rolling.get("l5_avg"),
        "l20_hit_rate": inputs.rolling.get("l20_hit_rate"),
        "l10_hit_rate": inputs.rolling.get("l10_hit_rate"),
        "l5_hit_rate": inputs.rolling.get("l5_hit_rate"),
        "minutes_per_game": inputs.minutes_profile.get("season_avg_min"),
        "projected_minutes": projected_minutes,
        "usage_pct": (inputs.usage_profile or {}).get("l10_usage_pct") or (inputs.usage_profile or {}).get("season_usage_pct"),
        "starter": inputs.minutes_profile.get("rotation_role") == "Starter",
        "rotation_role": inputs.minutes_profile.get("rotation_role"),
        "teammate_usage_bump_pct": inputs.teammate_usage_bump_pct,
        "teammate_minutes_bump_pct": inputs.teammate_minutes_bump_pct,
        "opponent_pace_rank": inputs.opponent_pace_rank,
        "opponent_def_rating_rank": inputs.opponent_def_rating_rank,
        "dvp_rank": inputs.dvp_rank,
        "game_total": inputs.game_total,
        "spread": inputs.spread,
        "home_away": "Home" if inputs.is_home else ("Away" if inputs.is_home is not None else None),
        "rest_days": inputs.rest_days,
        "back_to_back": inputs.is_b2b,
        "h2h_hit_rate": (inputs.h2h or {}).get("h2h_hit_rate"),
        "manual_matchup_adjustment": inputs.manual_matchup_adjustment,
        "context_lean": inputs.context_lean,
        "context_note": inputs.context_note,
        "lineup_confirmed": inputs.lineup_confirmed,
    }

    return PredictionResult(
        raw_predicted_prob=raw_prob,
        predicted_prob=calibrated_prob,
        projected_mean=projected_mean,
        projected_std=projected_std,
        base_per_min_rate=blended_rate,
        adjustments=adjustments,
        total_adjustment=total_adjustment,
        confidence=confidence,
        confidence_score=score,
        breakdown=breakdown,
    )
