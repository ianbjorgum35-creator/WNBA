const $ = (id) => document.getElementById(id);

let PROP_LABELS = {}; // key -> label
let lastRecord = null;
let selectedPendingId = null;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("session expired, redirecting to login");
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

function showLog(el, text) {
  el.hidden = false;
  el.textContent = text;
}

function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
async function init() {
  const cfg = await api("/api/config");
  const teamSelects = [$("playerTeam"), $("opponentTeam")];
  for (const sel of teamSelects) {
    sel.innerHTML = cfg.teams.map((t) => `<option value="${t}">${t}</option>`).join("");
  }
  $("opponentTeam").selectedIndex = 1;

  $("propType").innerHTML = cfg.prop_types.map((p) => `<option value="${p.key}">${p.label}</option>`).join("");
  for (const p of cfg.prop_types) PROP_LABELS[p.key] = p.label;

  $("direction").innerHTML = cfg.directions.map((d) => `<option value="${d}">${d}</option>`).join("");

  refreshDriveStatus();
  refreshPending();
}

$("manualAdj").addEventListener("input", (e) => {
  $("manualAdjVal").textContent = (parseFloat(e.target.value) * 100).toFixed(0) + "%";
});
$("contextLean").addEventListener("input", (e) => {
  $("contextLeanVal").textContent = (parseFloat(e.target.value) * 100).toFixed(0) + "%";
});

// ---------------------------------------------------------------------
// Auto-fetch
// ---------------------------------------------------------------------
function applyFetchedFields(fields) {
  const map = {
    season_avg: "seasonAvg", l20_avg: "l20Avg", l10_avg: "l10Avg", l5_avg: "l5Avg",
    l20_hit: "l20Hit", l10_hit: "l10Hit", l5_hit: "l5Hit",
    min_per_game: "minPerGame", proj_minutes: "projMinutes", usage_pct: "usagePct",
    rotation_role: "rotationRole",
    pace_rank: "paceRank", def_rank: "defRank", dvp_rank: "dvpRank", h2h_hit: "h2hHit",
    rest_days: "restDays",
  };
  for (const [k, id] of Object.entries(map)) {
    if (fields[k] !== undefined && fields[k] !== null) $(id).value = fields[k];
  }
  if (fields.starter !== undefined) $("starter").checked = !!fields.starter;
  if (fields.home_away) $("homeAway").value = fields.home_away;
  if (fields.is_b2b !== undefined) $("isB2b").checked = !!fields.is_b2b;
}

$("btnFetch").addEventListener("click", async () => {
  const btn = $("btnFetch");
  btn.disabled = true;
  showLog($("fetchLog"), "Fetching live data... (unofficial endpoints -- if this fails, just fill fields in by hand)");
  try {
    const body = {
      player_name: $("playerName").value,
      player_team: $("playerTeam").value,
      opponent_team: $("opponentTeam").value,
      prop_key: $("propType").value,
      line: parseFloat($("line").value) || 0,
      direction: $("direction").value,
    };
    const res = await api("/api/autofetch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    applyFetchedFields(res.fields);
    showLog($("fetchLog"), res.messages.join("\n") + "\n\nDone. Review/edit any fields above, then click Run Prediction.");
  } catch (e) {
    showLog($("fetchLog"), "Fetch failed:\n" + e.message);
  } finally {
    btn.disabled = false;
  }
});

$("btnDiagnostics").addEventListener("click", async () => {
  showLog($("fetchLog"), "Running diagnostics...");
  try {
    const params = new URLSearchParams({
      player_name: $("playerName").value || "",
      player_team_name: $("playerTeam").value || "",
    });
    const report = await api("/api/diagnostics?" + params.toString());
    const failed = Object.entries(report).filter(([, v]) => !v.ok);
    let text = JSON.stringify(report, null, 2);
    text += failed.length
      ? `\n\n${failed.length} check(s) failed: ${failed.map(([k]) => k).join(", ")}`
      : "\n\nAll checks passed.";
    showLog($("fetchLog"), text);
  } catch (e) {
    showLog($("fetchLog"), "Diagnostics failed:\n" + e.message);
  }
});

// ---------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------
function gatherPredictBody() {
  return {
    player_name: $("playerName").value,
    player_team: $("playerTeam").value,
    opponent_team: $("opponentTeam").value,
    prop_key: $("propType").value,
    direction: $("direction").value,
    line: parseFloat($("line").value) || 0,
    odds: parseFloat($("odds").value) || -110,
    game_total: parseFloat($("gameTotal").value) || null,
    spread: parseFloat($("spread").value) || 0,
    home_away: $("homeAway").value,
    rest_days: parseInt($("restDays").value) || 0,
    is_b2b: $("isB2b").checked,
    lineup_confirmed: $("lineupConfirmed").checked,
    season_avg: parseFloat($("seasonAvg").value) || null,
    l20_avg: parseFloat($("l20Avg").value) || null,
    l10_avg: parseFloat($("l10Avg").value) || null,
    l5_avg: parseFloat($("l5Avg").value) || null,
    l20_hit: parseFloat($("l20Hit").value) || null,
    l10_hit: parseFloat($("l10Hit").value) || null,
    l5_hit: parseFloat($("l5Hit").value) || null,
    min_per_game: parseFloat($("minPerGame").value) || null,
    proj_minutes: parseFloat($("projMinutes").value) || null,
    usage_pct: parseFloat($("usagePct").value) || null,
    starter: $("starter").checked,
    rotation_role: $("rotationRole").value,
    teammate_usage_bump: parseFloat($("teammateUsageBump").value) || null,
    teammate_minutes_bump: parseFloat($("teammateMinutesBump").value) || null,
    pace_rank: parseFloat($("paceRank").value) || null,
    def_rank: parseFloat($("defRank").value) || null,
    dvp_rank: parseFloat($("dvpRank").value) || null,
    h2h_hit: parseFloat($("h2hHit").value) || null,
    manual_adj: parseFloat($("manualAdj").value) || 0,
    context_lean: parseFloat($("contextLean").value) || 0,
    context_note: $("contextNote").value,
  };
}

function renderResult(res) {
  const card = $("resultCard");
  card.hidden = false;
  const edgeClass = res.edge >= 0 ? "edge-pos" : "edge-neg";
  card.innerHTML = `
    <div class="big-prob">${fmtPct(res.predicted_prob)}</div>
    <div class="result-row"><span>Raw model probability</span><span>${fmtPct(res.raw_predicted_prob)}</span></div>
    <div class="result-row"><span>Sportsbook implied probability</span><span>${fmtPct(res.implied_prob)}</span></div>
    <div class="result-row"><span>Edge (model - market)</span><span class="${edgeClass}">${(res.edge >= 0 ? "+" : "") + fmtPct(res.edge)}</span></div>
    <div class="result-row"><span>Projected mean (std)</span><span>${res.projected_mean.toFixed(2)} (${res.projected_std.toFixed(2)})</span></div>
    <div class="result-row"><span>Confidence</span><span class="pill">${res.confidence} (${res.confidence_score.toFixed(2)})</span></div>
    <div class="result-row"><span>Quarter-Kelly stake</span><span>${fmtPct(res.kelly_quarter)} of bankroll</span></div>
    <details style="margin-top:0.6rem;">
      <summary>Adjustment breakdown</summary>
      <table>${Object.entries(res.adjustments).map(([k, v]) =>
        `<tr><td>${k}</td><td>${(v >= 0 ? "+" : "") + fmtPct(v)}</td></tr>`).join("")}
        <tr><td><strong>TOTAL</strong></td><td><strong>${(res.total_adjustment >= 0 ? "+" : "") + fmtPct(res.total_adjustment)}</strong></td></tr>
      </table>
    </details>
  `;
}

$("btnPredict").addEventListener("click", async () => {
  const btn = $("btnPredict");
  btn.disabled = true;
  try {
    const body = gatherPredictBody();
    const res = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    renderResult(res);
    lastRecord = res.record;
  } catch (e) {
    alert("Prediction failed: " + e.message);
  } finally {
    btn.disabled = false;
  }
});

$("btnLog").addEventListener("click", async () => {
  if (!lastRecord) {
    alert("Run a prediction first.");
    return;
  }
  try {
    const res = await api("/api/log", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(lastRecord),
    });
    alert(`Logged as bet_id=${res.bet_id}. Come back after the game to record the outcome below.`);
    refreshPending();
  } catch (e) {
    alert("Logging failed: " + e.message);
  }
});

// ---------------------------------------------------------------------
// Pending bets / outcomes
// ---------------------------------------------------------------------
async function refreshPending() {
  const list = await api("/api/pending");
  const container = $("pendingList");
  if (!list.length) {
    container.innerHTML = '<p class="hint">No pending bets.</p>';
    selectedPendingId = null;
    return;
  }
  container.innerHTML = `<table><tbody>${list.map((b) => `
    <tr class="pending-row" data-id="${b.bet_id}">
      <td>${b.player_name}</td>
      <td>${b.direction} ${b.line} ${b.prop_type}</td>
      <td>${b.player_team} vs ${b.opponent_team}</td>
    </tr>`).join("")}</tbody></table>`;
  container.querySelectorAll(".pending-row").forEach((row) => {
    row.addEventListener("click", () => {
      container.querySelectorAll(".pending-row").forEach((r) => r.classList.remove("selected"));
      row.classList.add("selected");
      selectedPendingId = row.dataset.id;
    });
  });
}

$("btnRefreshPending").addEventListener("click", refreshPending);

$("btnRecordOutcome").addEventListener("click", async () => {
  if (!selectedPendingId) {
    alert("Select a pending bet first (click Refresh Pending List, then click a row).");
    return;
  }
  const actual = parseFloat($("actualValue").value);
  if (Number.isNaN(actual)) {
    alert("Enter the actual stat value.");
    return;
  }
  try {
    const body = {
      bet_id: selectedPendingId,
      actual_stat_value: actual,
      closing_odds: parseFloat($("closingOdds").value) || null,
    };
    const res = await api("/api/record_outcome", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    showLog($("outcomeLog"), `Recorded outcome for ${selectedPendingId}: hit=${res.updated.hit}, clv_pct=${res.updated.clv_pct}`);
    refreshPending();
  } catch (e) {
    showLog($("outcomeLog"), "Recording failed:\n" + e.message);
  }
});

// ---------------------------------------------------------------------
// Model learning
// ---------------------------------------------------------------------
$("btnRecalibrate").addEventListener("click", async () => {
  showLog($("learningLog"), "Recalibrating...");
  try {
    const res = await api("/api/recalibrate", { method: "POST" });
    showLog($("learningLog"), JSON.stringify(res, null, 2));
  } catch (e) {
    showLog($("learningLog"), "Recalibration failed:\n" + e.message);
  }
});

$("btnCalibration").addEventListener("click", async () => {
  try {
    const res = await api("/api/calibration");
    showLog($("learningLog"), JSON.stringify(res, null, 2));
  } catch (e) {
    showLog($("learningLog"), "Failed:\n" + e.message);
  }
});

// ---------------------------------------------------------------------
// Drive sync
// ---------------------------------------------------------------------
async function refreshDriveStatus() {
  try {
    const st = await api("/api/drive/status");
    const pill = $("driveStatus");
    pill.textContent = "Drive: " + st.message;
    pill.className = "pill " + (st.connected ? "ok" : st.enabled ? "warn" : "bad");
  } catch (e) {
    $("driveStatus").textContent = "Drive: unknown";
  }
}

$("btnDrivePull").addEventListener("click", async () => {
  showLog($("driveLog"), "Pulling from Drive...");
  const res = await api("/api/drive/pull", { method: "POST" });
  showLog($("driveLog"), JSON.stringify(res, null, 2));
  refreshDriveStatus();
  refreshPending();
});

$("btnDrivePush").addEventListener("click", async () => {
  showLog($("driveLog"), "Pushing to Drive...");
  const res = await api("/api/drive/push", { method: "POST" });
  showLog($("driveLog"), JSON.stringify(res, null, 2));
  refreshDriveStatus();
});

init();
