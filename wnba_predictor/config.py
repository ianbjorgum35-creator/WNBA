"""Static configuration: league structure, prop definitions, and model defaults.

Team roster reflects the 2026 WNBA season (15 teams: the 13 from 2025 plus
the Toronto Tempo and Portland Fire expansion franchises). Update
WNBA_TEAMS if the league realigns again.
"""

from dataclasses import dataclass, field

WNBA_TEAMS = [
    {"name": "Atlanta Dream", "abbr": "ATL"},
    {"name": "Chicago Sky", "abbr": "CHI"},
    {"name": "Connecticut Sun", "abbr": "CON"},
    {"name": "Dallas Wings", "abbr": "DAL"},
    {"name": "Golden State Valkyries", "abbr": "GSV"},
    {"name": "Indiana Fever", "abbr": "IND"},
    {"name": "Las Vegas Aces", "abbr": "LVA"},
    {"name": "Los Angeles Sparks", "abbr": "LAS"},
    {"name": "Minnesota Lynx", "abbr": "MIN"},
    {"name": "New York Liberty", "abbr": "NYL"},
    {"name": "Phoenix Mercury", "abbr": "PHX"},
    {"name": "Portland Fire", "abbr": "POR"},
    {"name": "Seattle Storm", "abbr": "SEA"},
    {"name": "Toronto Tempo", "abbr": "TOR"},
    {"name": "Washington Mystics", "abbr": "WAS"},
]

TEAM_NAMES = [t["name"] for t in WNBA_TEAMS]


@dataclass(frozen=True)
class PropType:
    key: str
    label: str
    # boxscore columns (as returned by data_sources game logs) summed to build the stat
    columns: tuple
    # "normal" for high-count continuous-ish stats, "poisson" for low-count discrete stats
    distribution: str = "normal"


PROP_TYPES = [
    PropType("PTS", "Points", ("PTS",), "normal"),
    PropType("REB", "Rebounds", ("REB",), "normal"),
    PropType("AST", "Assists", ("AST",), "normal"),
    PropType("PRA", "Pts + Reb + Ast", ("PTS", "REB", "AST"), "normal"),
    PropType("PR", "Pts + Reb", ("PTS", "REB"), "normal"),
    PropType("PA", "Pts + Ast", ("PTS", "AST"), "normal"),
    PropType("RA", "Reb + Ast", ("REB", "AST"), "normal"),
    PropType("FG3M", "3-Pointers Made", ("FG3M",), "poisson"),
    PropType("STL", "Steals", ("STL",), "poisson"),
    PropType("BLK", "Blocks", ("BLK",), "poisson"),
    PropType("STOCKS", "Steals + Blocks", ("STL", "BLK"), "poisson"),
    PropType("TOV", "Turnovers", ("TOV",), "poisson"),
]

PROP_TYPES_BY_KEY = {p.key: p for p in PROP_TYPES}

DIRECTIONS = ["Over", "Under"]

# Weighted blend of recency windows used to build the base per-minute rate.
# These are the weights the CLV/outcome learning loop (clv.py) is allowed to
# adjust over time; this dict is only the cold-start prior.
DEFAULT_BLEND_WEIGHTS = {
    "season": 0.15,
    "l20": 0.25,
    "l10": 0.30,
    "l5": 0.30,
}

# Max absolute percentage adjustment any single contextual factor may apply
# to the projected mean, to keep any one input from dominating the model.
ADJUSTMENT_CAPS = {
    "pace": 0.08,
    "opp_defense": 0.10,
    "dvp": 0.10,
    "game_environment": 0.08,
    "home_away": 0.03,
    "rest": 0.04,
    "b2b": 0.05,
    "teammate_usage_bump": 0.15,
    "teammate_minutes_bump": 0.10,
    "manual_matchup": 0.15,
    "context_lean": 0.10,
}

# When the lineup isn't confirmed, we inflate variance (pull the predicted
# probability back toward 50%) to reflect minutes/role uncertainty.
UNCONFIRMED_LINEUP_VARIANCE_INFLATION = 1.35

DATA_DIR = "data"
BET_LOG_PATH = f"{DATA_DIR}/bet_log.csv"
LEARNED_WEIGHTS_PATH = f"{DATA_DIR}/learned_weights.json"
CACHE_DIR = f"{DATA_DIR}/cache"

BET_LOG_COLUMNS = [
    "bet_id",
    "logged_at",
    "player_name",
    "player_team",
    "opponent_team",
    "prop_type",
    "direction",
    "line",
    "odds_open",
    "implied_prob_open",
    "predicted_prob",
    "raw_predicted_prob",
    "edge",
    "season_hit_rate",
    "l20_hit_rate",
    "l10_hit_rate",
    "l5_hit_rate",
    "blend_weights_json",
    "adjustments_json",
    "breakdown_json",
    "confidence",
    "lineup_confirmed",
    "manual_matchup_adjustment",
    "context_lean",
    "closing_odds",
    "implied_prob_close",
    "clv_pct",
    "actual_stat_value",
    "hit",
    "outcome_logged_at",
    "notes",
]
