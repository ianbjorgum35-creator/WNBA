"""ipywidgets dropdown UI for Colab/Jupyter.

Design: an "Auto-Fetch Data" button attempts to pull every stat from the
live data sources and populate the editable fields below it; a "Run
Prediction" button always runs off whatever is currently in those fields, so
a broken/rate-limited endpoint never blocks you -- just type the number in
by hand and keep going. "Log This Prediction" appends to the CLV/outcome
log, and the bottom section resolves past bets and re-fits the calibrator.
"""

import datetime as dt
import json
import traceback

import ipywidgets as widgets
from IPython.display import clear_output, display

from . import clv, config, data_sources, matchup, projection, stats_engine
from .odds import edge as edge_fn
from .odds import implied_prob_from_american, kelly_stake

TEAM_NAMES = config.TEAM_NAMES
TEAM_ABBR = {t["name"]: t["abbr"] for t in config.WNBA_TEAMS}
PROP_LABELS = {p.label: p.key for p in config.PROP_TYPES}


def _style():
    return {"description_width": "160px"}


def _layout(width="420px"):
    return widgets.Layout(width=width)


class PredictorUI:
    def __init__(self, bet_log_path: str = config.BET_LOG_PATH,
                 weights_path: str = config.LEARNED_WEIGHTS_PATH):
        self.bet_log_path = bet_log_path
        self.weights_path = weights_path
        self._last_result = None
        self._last_record = None
        self._build_widgets()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self):
        s, lay = _style(), _layout()

        self.w_player_name = widgets.Text(description="Player Name", style=s, layout=lay)
        self.w_player_team = widgets.Dropdown(options=TEAM_NAMES, description="Player Team", style=s, layout=lay)
        self.w_opponent_team = widgets.Dropdown(options=TEAM_NAMES, description="Opponent Team",
                                                 value=TEAM_NAMES[1], style=s, layout=lay)
        self.w_prop_type = widgets.Dropdown(options=list(PROP_LABELS.keys()), description="Prop Type",
                                             style=s, layout=lay)
        self.w_direction = widgets.Dropdown(options=config.DIRECTIONS, description="Over/Under", style=s, layout=lay)
        self.w_line = widgets.FloatText(description="Line", style=s, layout=lay)
        self.w_odds = widgets.FloatText(description="Odds (American)", value=-110, style=s, layout=lay)

        self.w_game_total = widgets.FloatText(description="Game Total", style=s, layout=lay)
        self.w_spread = widgets.FloatText(description="Spread (team's own)", style=s, layout=lay)
        self.w_home_away = widgets.Dropdown(options=["Home", "Away"], description="Home/Away", style=s, layout=lay)
        self.w_rest_days = widgets.IntText(description="Rest Days", value=1, style=s, layout=lay)
        self.w_is_b2b = widgets.Checkbox(description="Back-to-Back?", value=False, style=s)
        self.w_lineup_confirmed = widgets.Checkbox(description="Lineup Confirmed?", value=True, style=s)

        # Auto-fetched / manually-overridable stat fields
        self.w_season_avg = widgets.FloatText(description="Season Average", style=s, layout=lay)
        self.w_l20_avg = widgets.FloatText(description="L20 Average", style=s, layout=lay)
        self.w_l10_avg = widgets.FloatText(description="L10 Average", style=s, layout=lay)
        self.w_l5_avg = widgets.FloatText(description="L5 Average", style=s, layout=lay)
        self.w_l20_hit = widgets.FloatText(description="L20 Hit Rate", style=s, layout=lay)
        self.w_l10_hit = widgets.FloatText(description="L10 Hit Rate", style=s, layout=lay)
        self.w_l5_hit = widgets.FloatText(description="L5 Hit Rate", style=s, layout=lay)
        self.w_min_per_game = widgets.FloatText(description="Minutes/Game", style=s, layout=lay)
        self.w_proj_minutes = widgets.FloatText(description="Projected Minutes", style=s, layout=lay)
        self.w_usage_pct = widgets.FloatText(description="Usage %", style=s, layout=lay)
        self.w_starter = widgets.Checkbox(description="Starter?", value=False, style=s)
        self.w_rotation_role = widgets.Text(description="Rotation Role", style=s, layout=lay)
        self.w_teammate_usage_bump = widgets.FloatText(description="Teammate Usage Bump", style=s, layout=lay)
        self.w_teammate_minutes_bump = widgets.FloatText(description="Teammate Minutes Bump", style=s, layout=lay)

        self.w_pace_rank = widgets.FloatText(description="Opp Pace Rank", style=s, layout=lay)
        self.w_def_rank = widgets.FloatText(description="Opp Defense Rank", style=s, layout=lay)
        self.w_dvp_rank = widgets.FloatText(description="DvP Rank", style=s, layout=lay)
        self.w_h2h_hit = widgets.FloatText(description="H2H Hit Rate", style=s, layout=lay)

        self.w_manual_adj = widgets.FloatSlider(description="Manual Matchup Adj", min=-0.15, max=0.15,
                                                 step=0.01, value=0.0, style=s, layout=lay)
        self.w_context_lean = widgets.FloatSlider(description="Context Lean", min=-0.10, max=0.10,
                                                   step=0.01, value=0.0, style=s, layout=lay)
        self.w_context_note = widgets.Text(description="Context Note", style=s, layout=lay)

        self.btn_fetch = widgets.Button(description="Auto-Fetch Data", button_style="info")
        self.btn_diagnostics = widgets.Button(description="Run Diagnostics")
        self.btn_predict = widgets.Button(description="Run Prediction", button_style="success")
        self.btn_log = widgets.Button(description="Log This Prediction", button_style="warning")
        self.out_fetch = widgets.Output()
        self.out_predict = widgets.Output()

        self.btn_fetch.on_click(self._on_fetch)
        self.btn_diagnostics.on_click(self._on_diagnostics)
        self.btn_predict.on_click(self._on_predict)
        self.btn_log.on_click(self._on_log)

        # --- Outcome recording section ---
        self.w_pending_bets = widgets.Dropdown(options=[], description="Pending Bet", style=s, layout=lay)
        self.btn_refresh_pending = widgets.Button(description="Refresh Pending List")
        self.w_actual_value = widgets.FloatText(description="Actual Stat Value", style=s, layout=lay)
        self.w_closing_odds = widgets.FloatText(description="Closing Odds", style=s, layout=lay)
        self.btn_record_outcome = widgets.Button(description="Record Outcome", button_style="warning")
        self.out_outcome = widgets.Output()
        self.btn_refresh_pending.on_click(self._on_refresh_pending)
        self.btn_record_outcome.on_click(self._on_record_outcome)

        # --- Recalibration section ---
        self.btn_recalibrate = widgets.Button(description="Recalibrate Model", button_style="danger")
        self.out_recalibrate = widgets.Output()
        self.btn_recalibrate.on_click(self._on_recalibrate)

    def display(self):
        input_box = widgets.VBox([
            widgets.HTML("<h3>Player & Prop</h3>"),
            self.w_player_name, self.w_player_team, self.w_opponent_team,
            self.w_prop_type, self.w_direction, self.w_line, self.w_odds,
            widgets.HTML("<h3>Game Environment</h3>"),
            self.w_game_total, self.w_spread, self.w_home_away,
            self.w_rest_days, self.w_is_b2b, self.w_lineup_confirmed,
            widgets.HTML("<h3>Player Stats (auto-fetched, edit as needed)</h3>"),
            self.w_season_avg, self.w_l20_avg, self.w_l10_avg, self.w_l5_avg,
            self.w_l20_hit, self.w_l10_hit, self.w_l5_hit,
            self.w_min_per_game, self.w_proj_minutes, self.w_usage_pct,
            self.w_starter, self.w_rotation_role,
            self.w_teammate_usage_bump, self.w_teammate_minutes_bump,
            widgets.HTML("<h3>Matchup (auto-fetched, edit as needed)</h3>"),
            self.w_pace_rank, self.w_def_rank, self.w_dvp_rank, self.w_h2h_hit,
            widgets.HTML("<h3>Manual Overrides</h3>"),
            self.w_manual_adj, self.w_context_lean, self.w_context_note,
            widgets.HBox([self.btn_fetch, self.btn_diagnostics, self.btn_predict, self.btn_log]),
            self.out_fetch, self.out_predict,
            widgets.HTML("<h3>Record Outcome for a Past Bet</h3>"),
            widgets.HBox([self.w_pending_bets, self.btn_refresh_pending]),
            self.w_actual_value, self.w_closing_odds, self.btn_record_outcome, self.out_outcome,
            widgets.HTML("<h3>Model Learning</h3>"),
            self.btn_recalibrate, self.out_recalibrate,
        ])
        display(input_box)

    # ------------------------------------------------------------------
    # Auto-fetch
    # ------------------------------------------------------------------
    def _on_fetch(self, _btn):
        with self.out_fetch:
            clear_output()
            print("Fetching live data... (unofficial endpoints -- if this fails, just fill the fields in by hand)")
            try:
                self._fetch_player_data()
            except Exception:
                print("Player stat fetch failed:")
                traceback.print_exc()
            try:
                self._fetch_matchup_data()
            except Exception:
                print("Matchup data fetch failed:")
                traceback.print_exc()
            print("Done. Review/edit any fields above, then click Run Prediction.")

    def _on_diagnostics(self, _btn):
        with self.out_fetch:
            clear_output()
            report = data_sources.diagnostics(
                player_name=self.w_player_name.value or None,
                player_team_name=self.w_player_team.value or None,
            )
            failed = [name for name, entry in report.items() if not entry["ok"]]
            if failed:
                print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}. "
                      "Share this output if you need help fixing it -- it pinpoints exactly "
                      "which host/step is broken and why.")
            else:
                print("\nAll checks passed. If Auto-Fetch still isn't populating fields, "
                      "something more specific to this player/team is failing -- try Auto-Fetch "
                      "again and check the messages it prints.")

    def _fetch_player_data(self):
        gamelog_res = None
        source_used = "stats_wnba"
        player_res = data_sources.get_player_id(self.w_player_name.value)
        if player_res.success:
            gamelog_res = data_sources.get_player_gamelog(player_res.data)
            if not gamelog_res.success:
                print(f"  stats.wnba.com game log fetch failed: {gamelog_res.message}")
        else:
            print(f"  stats.wnba.com player lookup failed: {player_res.message}")

        espn_opponent_abbr = None
        if gamelog_res is None or not gamelog_res.success:
            print("  Trying ESPN as a fallback source...")
            espn_team_res = data_sources.resolve_espn_team_abbr(self.w_player_team.value)
            if not espn_team_res.success:
                print(f"  Could not resolve ESPN team abbreviation: {espn_team_res.message}")
                return
            espn_id_res = data_sources.get_espn_player_id(self.w_player_name.value, espn_team_res.data)
            if not espn_id_res.success:
                print(f"  ESPN player lookup also failed: {espn_id_res.message}")
                return
            gamelog_res = data_sources.get_espn_player_gamelog(espn_id_res.data)
            if not gamelog_res.success:
                print(f"  ESPN game log fetch also failed: {gamelog_res.message}")
                return
            source_used = "espn"
            espn_opp_res = data_sources.resolve_espn_team_abbr(self.w_opponent_team.value)
            espn_opponent_abbr = espn_opp_res.data if espn_opp_res.success else None
            print("  Loaded game log from ESPN fallback.")

        df = stats_engine.normalize_gamelog(gamelog_res.data)
        self._player_gamelog = df
        prop_key = PROP_LABELS[self.w_prop_type.value]
        rolling = stats_engine.rolling_windows(df, prop_key, self.w_line.value, self.w_direction.value)
        minutes = stats_engine.minutes_profile(df)
        usage = stats_engine.usage_profile(df)
        self._rolling, self._minutes_profile, self._usage_profile = rolling, minutes, usage

        self.w_season_avg.value = rolling.get("season_avg") or 0.0
        self.w_l20_avg.value = rolling.get("l20_avg") or 0.0
        self.w_l10_avg.value = rolling.get("l10_avg") or 0.0
        self.w_l5_avg.value = rolling.get("l5_avg") or 0.0
        self.w_l20_hit.value = rolling.get("l20_hit_rate") or 0.0
        self.w_l10_hit.value = rolling.get("l10_hit_rate") or 0.0
        self.w_l5_hit.value = rolling.get("l5_hit_rate") or 0.0
        self.w_min_per_game.value = minutes.get("season_avg_min") or 0.0
        self.w_proj_minutes.value = minutes.get("l5_avg_min") or minutes.get("season_avg_min") or 0.0
        self.w_usage_pct.value = usage.get("l10_usage_pct") or usage.get("season_usage_pct") or 0.0
        self.w_starter.value = minutes.get("rotation_role") == "Starter"
        self.w_rotation_role.value = minutes.get("rotation_role") or ""
        print(f"  Player stats loaded from {len(df)} games.")

        if source_used == "espn":
            espn_own_res = data_sources.resolve_espn_team_abbr(self.w_player_team.value)
            own_abbr = espn_own_res.data if espn_own_res.success else TEAM_ABBR[self.w_player_team.value]
            opp_abbr = espn_opponent_abbr or TEAM_ABBR[self.w_opponent_team.value]
        else:
            own_abbr = TEAM_ABBR[self.w_player_team.value]
            opp_abbr = TEAM_ABBR[self.w_opponent_team.value]

        h2h = matchup.h2h_hit_rate(
            df, own_abbr, opp_abbr,
            prop_key, self.w_line.value, self.w_direction.value,
        )
        if h2h.get("h2h_hit_rate") is not None:
            self.w_h2h_hit.value = h2h["h2h_hit_rate"]
        print(f"  H2H vs {self.w_opponent_team.value}: {h2h.get('h2h_games', 0)} games found.")

    def _fetch_matchup_data(self):
        ranks_res = data_sources.get_team_ranks()
        used_fallback = False
        if not ranks_res.success:
            print(f"  stats.wnba.com team ranks fetch failed: {ranks_res.message}")
            print("  Trying ESPN standings as a fallback (pace becomes a rough proxy -- "
                  "combined scoring, not true possession-based pace)...")
            ranks_res = data_sources.get_espn_team_ranks_fallback()
            used_fallback = True
            if not ranks_res.success:
                print(f"  ESPN standings fallback also failed: {ranks_res.message}")
                print("  Fill Opp Pace Rank / Opp Defense Rank in by hand for now.")

        if ranks_res.success:
            entry = matchup.team_pace_and_defense(ranks_res.data, self.w_opponent_team.value)
            if entry.get("pace_rank") is not None:
                self.w_pace_rank.value = entry["pace_rank"]
            if entry.get("def_rating_rank") is not None:
                self.w_def_rank.value = entry["def_rating_rank"]
            note = " (ESPN standings proxy, not stats.wnba.com's real pace metric)" if used_fallback else ""
            print(f"  Opponent ranks loaded{note}: {entry}")
            print("  DvP rank has no reliable single-endpoint source -- leaving as manual entry "
                  "(use Opponent Defense Rank as a proxy, or fill in from your own research).")

        own_espn_res = data_sources.resolve_espn_team_abbr(self.w_player_team.value)
        opp_espn_res = data_sources.resolve_espn_team_abbr(self.w_opponent_team.value)
        if not (own_espn_res.success and opp_espn_res.success):
            print("  Could not resolve ESPN team abbreviations -- leaving Home/Away, "
                  "Rest Days, and Back-to-Back as manual entry.")
            return
        context_res = data_sources.game_context(own_espn_res.data, opp_espn_res.data)
        if not context_res.success:
            print(f"  Game context (home/away, rest days) fetch failed: {context_res.message}")
            print("  Leaving Home/Away, Rest Days, and Back-to-Back as manual entry.")
            return
        ctx = context_res.data
        if ctx.get("is_home") is not None:
            self.w_home_away.value = "Home" if ctx["is_home"] else "Away"
        if ctx.get("rest_days") is not None:
            self.w_rest_days.value = ctx["rest_days"]
        if ctx.get("is_b2b") is not None:
            self.w_is_b2b.value = ctx["is_b2b"]
        print(f"  Game context loaded: {ctx['game_date']}, "
              f"{'Home' if ctx.get('is_home') else 'Away'}, rest_days={ctx.get('rest_days')}.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _build_inputs(self) -> projection.ProjectionInputs:
        prop_key = PROP_LABELS[self.w_prop_type.value]
        rolling = {
            "season_avg": self.w_season_avg.value or None,
            "l20_avg": self.w_l20_avg.value or None,
            "l10_avg": self.w_l10_avg.value or None,
            "l5_avg": self.w_l5_avg.value or None,
            "l20_hit_rate": self.w_l20_hit.value,
            "l10_hit_rate": self.w_l10_hit.value,
            "l5_hit_rate": self.w_l5_hit.value,
            "season_hit_rate": self.w_l20_hit.value,
            "season_games": getattr(self, "_rolling", {}).get("season_games", 10),
            "season_std": None, "l20_std": None, "l10_std": None,
        }
        minutes_profile = {
            "season_avg_min": self.w_min_per_game.value or None,
            "l10_avg_min": self.w_min_per_game.value or None,
            "l5_avg_min": self.w_proj_minutes.value or self.w_min_per_game.value or None,
            "rotation_role": self.w_rotation_role.value or ("Starter" if self.w_starter.value else "Rotation"),
        }
        usage_profile = {"l10_usage_pct": self.w_usage_pct.value or None}

        weights_res = clv.update_blend_weights(self.bet_log_path)
        blend_weights = weights_res.get("weights", config.DEFAULT_BLEND_WEIGHTS)

        return projection.ProjectionInputs(
            prop_key=prop_key,
            direction=self.w_direction.value,
            line=self.w_line.value,
            rolling=rolling,
            minutes_profile=minutes_profile,
            projected_minutes=self.w_proj_minutes.value or None,
            usage_profile=usage_profile,
            opponent_pace_rank=self.w_pace_rank.value or None,
            opponent_def_rating_rank=self.w_def_rank.value or None,
            dvp_rank=self.w_dvp_rank.value or None,
            game_total=self.w_game_total.value or None,
            spread=self.w_spread.value,
            is_home=(self.w_home_away.value == "Home"),
            rest_days=self.w_rest_days.value,
            is_b2b=self.w_is_b2b.value,
            h2h={"h2h_hit_rate": self.w_h2h_hit.value or None},
            teammate_usage_bump_pct=self.w_teammate_usage_bump.value or None,
            teammate_minutes_bump_pct=self.w_teammate_minutes_bump.value or None,
            manual_matchup_adjustment=self.w_manual_adj.value,
            context_lean=self.w_context_lean.value,
            context_note=self.w_context_note.value,
            lineup_confirmed=self.w_lineup_confirmed.value,
            blend_weights=blend_weights,
        )

    def _on_predict(self, _btn):
        with self.out_predict:
            clear_output()
            inputs = self._build_inputs()
            calibrator = clv.load_calibrator(self.weights_path)
            result = projection.run_prediction(inputs, calibrator=calibrator)
            self._last_result = result

            implied = implied_prob_from_american(self.w_odds.value)
            model_edge = edge_fn(result.predicted_prob, implied)
            kelly = kelly_stake(result.predicted_prob, self.w_odds.value)

            print(f"{self.w_player_name.value} ({self.w_player_team.value}) vs {self.w_opponent_team.value}")
            print(f"{self.w_direction.value} {self.w_line.value} {self.w_prop_type.value}  @ {self.w_odds.value}")
            print("-" * 60)
            print(f"Projected mean: {result.projected_mean:.2f}  (std {result.projected_std:.2f})")
            print(f"Raw model probability:        {result.raw_predicted_prob:.1%}")
            print(f"Calibrated probability:       {result.predicted_prob:.1%}")
            print(f"Sportsbook implied probability: {implied:.1%}")
            print(f"Edge (model - market):        {model_edge:+.1%}")
            print(f"Confidence: {result.confidence} (score {result.confidence_score:.2f})")
            print(f"Quarter-Kelly stake (informational only): {kelly.quarter_kelly:.1%} of bankroll")
            print("-" * 60)
            print("Adjustment breakdown:")
            for k, v in result.adjustments.items():
                print(f"  {k:>24s}: {v:+.1%}")
            print(f"  {'TOTAL':>24s}: {result.total_adjustment:+.1%}")

            self._last_record = {
                "player_name": self.w_player_name.value,
                "player_team": self.w_player_team.value,
                "opponent_team": self.w_opponent_team.value,
                "prop_type": self.w_prop_type.value,
                "direction": self.w_direction.value,
                "line": self.w_line.value,
                "odds_open": self.w_odds.value,
                "implied_prob_open": implied,
                "predicted_prob": result.predicted_prob,
                "raw_predicted_prob": result.raw_predicted_prob,
                "edge": model_edge,
                "season_hit_rate": inputs.rolling.get("season_hit_rate"),
                "l20_hit_rate": inputs.rolling.get("l20_hit_rate"),
                "l10_hit_rate": inputs.rolling.get("l10_hit_rate"),
                "l5_hit_rate": inputs.rolling.get("l5_hit_rate"),
                "blend_weights_json": json.dumps(inputs.blend_weights),
                "adjustments_json": json.dumps(result.adjustments),
                "breakdown_json": json.dumps(result.breakdown, default=str),
                "confidence": result.confidence,
                "lineup_confirmed": inputs.lineup_confirmed,
                "manual_matchup_adjustment": inputs.manual_matchup_adjustment,
                "context_lean": inputs.context_lean,
            }

    def _on_log(self, _btn):
        with self.out_predict:
            if self._last_record is None:
                print("Run a prediction first.")
                return
            bet_id = clv.log_bet(self._last_record, self.bet_log_path)
            print(f"\nLogged as bet_id={bet_id}. Come back after the game to record the outcome below.")

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------
    def _on_refresh_pending(self, _btn):
        with self.out_outcome:
            clear_output()
            df = clv.pending_bets(self.bet_log_path)
            if df.empty:
                self.w_pending_bets.options = []
                print("No pending bets.")
                return
            options = [
                (f"{row.bet_id}: {row.player_name} {row.direction} {row.line} {row.prop_type}", row.bet_id)
                for row in df.itertuples()
            ]
            self.w_pending_bets.options = options
            print(f"{len(options)} pending bet(s) loaded.")

    def _on_record_outcome(self, _btn):
        with self.out_outcome:
            clear_output()
            bet_id = self.w_pending_bets.value
            if not bet_id:
                print("Select a pending bet first (click Refresh Pending List).")
                return
            updated = clv.record_outcome(
                bet_id, self.w_actual_value.value,
                closing_odds=self.w_closing_odds.value or None,
                path=self.bet_log_path,
            )
            print(f"Recorded outcome for {bet_id}: hit={updated['hit']}, clv_pct={updated.get('clv_pct')}")

    # ------------------------------------------------------------------
    # Recalibration
    # ------------------------------------------------------------------
    def _on_recalibrate(self, _btn):
        with self.out_recalibrate:
            clear_output()
            report = clv.calibration_report(self.bet_log_path)
            print("Calibration report:", json.dumps(report, indent=2, default=str))
            fit = clv.fit_recalibration(self.bet_log_path, weights_path=self.weights_path)
            print("\nProbability recalibration:", json.dumps(fit, indent=2, default=str))
            weights = clv.update_blend_weights(self.bet_log_path)
            print("\nBlend weight update:", json.dumps(weights, indent=2, default=str))


def launch(bet_log_path: str = config.BET_LOG_PATH, weights_path: str = config.LEARNED_WEIGHTS_PATH) -> PredictorUI:
    ui = PredictorUI(bet_log_path=bet_log_path, weights_path=weights_path)
    ui.display()
    return ui
