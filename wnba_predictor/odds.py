"""American odds <-> implied probability, vig removal, edge, and CLV math."""

from dataclasses import dataclass
from typing import Optional


def implied_prob_from_american(odds: float) -> float:
    """Convert American odds (e.g. -110, +150) to raw implied probability."""
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def american_from_implied_prob(prob: float) -> float:
    """Inverse of implied_prob_from_american, for display/back-of-envelope use."""
    if not 0 < prob < 1:
        raise ValueError("prob must be in (0, 1)")
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


def devig_two_way(over_odds: float, under_odds: float) -> tuple:
    """Multiplicative no-vig de-vig for a two-way market.

    Returns (p_over_fair, p_under_fair) that sum to 1.
    """
    p_over = implied_prob_from_american(over_odds)
    p_under = implied_prob_from_american(under_odds)
    total = p_over + p_under
    return p_over / total, p_under / total


def edge(predicted_prob: float, implied_prob: float) -> float:
    """Positive edge means the model likes the side more than the market does."""
    return predicted_prob - implied_prob


def clv_percent(implied_prob_open: float, implied_prob_close: float) -> float:
    """Closing Line Value, expressed in implied-probability points.

    Positive CLV means the closing implied probability rose above what you
    paid for (the market moved toward you) -- i.e. you got a better number
    than the market eventually settled on.
    """
    return (implied_prob_close - implied_prob_open) * 100.0


@dataclass
class KellyResult:
    full_kelly: float
    quarter_kelly: float
    recommended_fraction: float


def kelly_stake(predicted_prob: float, american_odds: float, fraction: float = 0.25) -> KellyResult:
    """Informational fractional-Kelly stake sizing given model probability and price.

    Not a betting recommendation -- purely a sizing reference assuming the
    model's probability is correct. Negative values indicate no edge (b*p <= q).
    """
    odds = float(american_odds)
    b = (odds / 100.0) if odds > 0 else (100.0 / -odds)
    p = predicted_prob
    q = 1 - p
    full = (b * p - q) / b if b > 0 else 0.0
    full = max(full, 0.0)
    return KellyResult(full_kelly=full, quarter_kelly=full * fraction, recommended_fraction=fraction)


def implied_team_totals(game_total: float, spread: float) -> tuple:
    """Given a game total and a team's own spread (negative = favorite),
    return (this_team_implied_total, opponent_implied_total)."""
    this_team = game_total / 2.0 - spread / 2.0
    opponent = game_total - this_team
    return this_team, opponent
