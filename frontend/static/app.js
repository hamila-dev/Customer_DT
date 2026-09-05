// Customer Twin — simple vanilla-JS single-page app.
// No build step, no framework: fetch() against the FastAPI backend under
// /api/*, hash-based routing, and small render() functions per view.

const API = "/api";

const state = {
  customers: [],
  selectedCustomerId: null,
  modelAvailable: null, // null = unknown yet, true/false once known
};

const BASELINE_MC_TRIALS = 100;
const EVENTS_POLL_MS = 5000;
const EVENTS_FEED_CAP = 50;

let viewPollTimer = null;

function stopViewTimers() {
  if (viewPollTimer != null) {
    clearInterval(viewPollTimer);
    viewPollTimer = null;
  }
}

// ---------------------------------------------------------------- helpers

async function apiGet(path) {
  const res = await fetch(API + path);
  if (!res.ok) {
    const body = await safeJson(res);
    throw new ApiError(res.status, body?.detail || res.statusText);
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const errBody = await safeJson(res);
    throw new ApiError(res.status, errBody?.detail || res.statusText);
  }
  return res.json();
}

async function safeJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function riskBadge(level) {
  if (!level) return el("span", { class: "badge badge-neutral" }, "unscored");
  const cls = { HIGH: "badge-high", MEDIUM: "badge-medium", LOW: "badge-low" }[level] || "badge-neutral";
  return el("span", { class: `badge ${cls}` }, level);
}

function pct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function eventLabel(eventType) {
  return eventType.replace(/_/g, " ");
}

function customerLabel(c) {
  // The new dataset has no customer name - build a readable label from
  // policy_type/region_name/customer_id instead.
  const policy = c.policy_type || "";
  const region = c.region_name || "";
  const parts = [policy, region].filter(Boolean).join(" · ");
  return parts ? `${parts} (${c.customer_id})` : c.customer_id;
}

function customerMatchesQuery(c, query) {
  if (!query) return true;
  const text = query.toLowerCase();
  return customerLabel(c).toLowerCase().includes(text) || String(c.customer_id).toLowerCase().includes(text);
}

function histogramEl(distributionSample, extraClass) {
  const buckets = new Array(20).fill(0);
  (distributionSample || []).forEach((v) => {
    const idx = Math.min(19, Math.max(0, Math.floor(v * 20)));
    buckets[idx] += 1;
  });
  const maxBucket = Math.max(...buckets, 1);
  return el(
    "div",
    { class: extraClass ? `hist-bars ${extraClass}` : "hist-bars" },
    buckets.map((count) => el("div", { class: "hist-bar", style: `height:${(count / maxBucket) * 100}%` }))
  );
}

/**
 * Baseline Monte Carlo for the Digital Twin view.
 *
 * The Monte Carlo engine perturbs numeric scenario parameters with
 * multiplicative Gaussian noise (mean=1.0, std=0.10 by default).
 * `premium_changed` + `change_pct: 0` would therefore stay 0 in every
 * trial and produce a degenerate (flat) distribution. Passing the
 * customer's actual `current_premium` lets that ±10% noise move a
 * non-zero dollar amount — sensitivity to normal premium fluctuation,
 * NOT a statistical confidence interval or margin of error on the model.
 * The simulate/monte-carlo endpoint clones Twin state and never persists.
 */
function fetchBaselineMonteCarlo(customerId, currentPremium) {
  return apiPost(`/customers/${customerId}/simulate/monte-carlo`, {
    scenario: "premium_changed",
    parameters: { current_premium: currentPremium },
    trials: BASELINE_MC_TRIALS,
  });
}

function fillStabilityPanel(panel, mc) {
  panel.innerHTML = "";
  panel.appendChild(el("div", { class: "stability-title" }, "Stability range (±10% premium)"));
  panel.appendChild(
    el("div", { class: "stability-stats" }, [
      el("div", {}, [el("div", { class: "small muted" }, "Mean"), el("div", { class: "stat-val" }, pct(mc.mean_churn_probability))]),
      el("div", {}, [el("div", { class: "small muted" }, "P10"), el("div", { class: "stat-val" }, pct(mc.p10_churn_probability))]),
      el("div", {}, [el("div", { class: "small muted" }, "P90"), el("div", { class: "stat-val" }, pct(mc.p90_churn_probability))]),
      el("div", {}, [el("div", { class: "small muted" }, "Std"), el("div", { class: "stat-val" }, pct(mc.std_dev))]),
    ])
  );
  panel.appendChild(histogramEl(mc.distribution_sample, "compact"));
  panel.appendChild(
    el("div", { class: "small muted", style: "margin-top:8px;" }, "Simulated premium noise, not a confidence interval.")
  );
}

function recMcLineText(pointProbability, mc) {
  return `${pct(pointProbability)} · MC ${pct(mc.p10_churn_probability)}–${pct(mc.p90_churn_probability)}`;
}

// ---------------------------------------------------------------- topbar

async function refreshTopbar() {
  try {
    const status = await apiGet("/event-generator/status");
    const dot = document.getElementById("gen-dot");
    const label = document.getElementById("gen-label");
    dot.className = "dot " + (status.running ? "dot-on" : "dot-off");
    label.textContent = status.running
      ? `Source on · ${status.events_generated}`
      : "Source off";
  } catch {
    /* non-fatal */
  }

  // Reuse whatever the current view already learned about model
  // availability rather than issuing a second /api/customers request on
  // every navigation (that endpoint scores every customer, so avoid
  // calling it twice per page load).
  try {
    const pill = document.getElementById("model-pill");
    if (state.modelAvailable !== null) {
      pill.textContent = state.modelAvailable ? "Model on" : "No model";
      pill.className = "model-pill " + (state.modelAvailable ? "ok" : "missing");
    }
  } catch {
    /* non-fatal */
  }
}

// ---------------------------------------------------------------- router

const routes = {
  dashboard: { title: "Dashboard", render: renderDashboard },
  customers: { title: "Customers", render: renderCustomers },
  twin: { title: "Digital Twin", render: renderTwin },
  events: { title: "Events", render: renderEvents },
  simulation: { title: "Simulation", render: renderSimulation },
};

function currentRoute() {
  const hash = location.hash.replace(/^#\//, "");
  const [route, param] = hash.split("/");
  return { route: routes[route] ? route : "dashboard", param };
}

async function router() {
  stopViewTimers();
  const { route, param } = currentRoute();
  document.querySelectorAll(".nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  document.getElementById("page-title").textContent = routes[route].title;
  const view = document.getElementById("view");
  view.innerHTML = '<div class="spinner">Loading…</div>';
  try {
    await routes[route].render(view, param);
  } catch (err) {
    view.innerHTML = "";
    view.appendChild(
      el("div", { class: "card card-pad" }, [
        el("div", { class: "section-title" }, "Error"),
        el("div", { class: "small muted" }, err.message || String(err)),
      ])
    );
  }
  refreshTopbar();
}

window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", () => {
  if (!location.hash) location.hash = "#/dashboard";
  router();
});

// ---------------------------------------------------------------- dashboard

async function renderDashboard(view) {
  const summary = await apiGet("/dashboard/summary");
  state.modelAvailable = summary.model_available;
  view.innerHTML = "";

  const kpis = el("div", { class: "kpi-row" }, [
    kpiCard("Total", summary.total_customers, ""),
    kpiCard("High", summary.high_risk, "high"),
    kpiCard("Medium", summary.medium_risk, "medium"),
    kpiCard("Low", summary.low_risk, "low"),
  ]);
  view.appendChild(kpis);

  if (!summary.model_available) {
    view.appendChild(
      el("div", { class: "callout callout-warn", style: "margin-bottom:16px;" }, [
        "Model missing — add model/*.joblib.",
      ])
    );
  }

  const twoCol = el("div", { class: "two-col" });

  // High-risk customer list
  const highRiskCard = el("div", { class: "card card-pad" });
  highRiskCard.appendChild(el("div", { class: "section-title" }, "High risk"));
  if (!summary.high_risk_customers.length) {
    highRiskCard.appendChild(el("div", { class: "empty-state small" }, "None."));
  } else {
    const table = el("table", {}, [
      el("thead", {}, el("tr", {}, [el("th", {}, "Customer"), el("th", {}, "Churn")])),
    ]);
    const tbody = el("tbody");
    summary.high_risk_customers.forEach((c) => {
      const tr = el(
        "tr",
        { onclick: () => (location.hash = `#/twin/${c.customer_id}`) },
        [el("td", {}, customerLabel(c)), el("td", { class: "mono" }, pct(c.churn_probability))]
      );
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    highRiskCard.appendChild(table);
  }
  twoCol.appendChild(highRiskCard);

  // Recent events feed
  const eventsCard = el("div", { class: "card card-pad" });
  eventsCard.appendChild(el("div", { class: "section-title" }, "Recent events"));
  if (!summary.recent_events.length) {
    eventsCard.appendChild(el("div", { class: "empty-state small" }, "No events."));
  } else {
    const timeline = el("div", { class: "timeline" });
    summary.recent_events.forEach((e) => {
      timeline.appendChild(
        el("div", { class: "timeline-item" }, [
          el("div", { class: "timeline-dot" }),
          el("div", { class: "timeline-body" }, [
            el("div", {}, [el("b", {}, customerLabel(e)), ` — ${eventLabel(e.event_type)}`]),
            el("div", { class: "small muted" }, e.description || ""),
            el("div", { class: "timeline-time" }, fmtTime(e.occurred_at)),
          ]),
        ])
      );
    });
    eventsCard.appendChild(timeline);
  }
  twoCol.appendChild(eventsCard);

  view.appendChild(twoCol);
}

function kpiCard(label, value, cls) {
  return el("div", { class: `card kpi-card ${cls}` }, [
    el("div", { class: "kpi-value" }, String(value)),
    el("div", { class: "kpi-label" }, label),
  ]);
}

// ---------------------------------------------------------------- customers

async function renderCustomers(view) {
  const data = await apiGet("/customers");
  state.customers = data.customers;
  state.modelAvailable = data.customers.some((c) => c.risk_level !== null);
  view.innerHTML = "";

  const searchRow = el("div", { class: "row", style: "margin-bottom:12px;" }, [
    el("input", { type: "text", placeholder: "Search…", id: "customer-search", style: "width:280px;" }),
    el("span", { class: "small muted" }, `${data.total} customers`),
  ]);
  view.appendChild(searchRow);

  const card = el("div", { class: "card" });
  const table = el("table", {}, [
    el(
      "thead",
      {},
      el("tr", {}, [
        el("th", {}, "Customer"),
        el("th", {}, "Region"),
        el("th", {}, "Age"),
        el("th", {}, "Risk"),
        el("th", {}, "Churn"),
      ])
    ),
  ]);
  const tbody = el("tbody");
  table.appendChild(tbody);
  card.appendChild(table);
  view.appendChild(card);

  function renderRows(filterText) {
    tbody.innerHTML = "";
    const filtered = data.customers.filter((c) => customerMatchesQuery(c, filterText));
    if (!filtered.length) {
      tbody.appendChild(el("tr", {}, el("td", { colspan: "5", class: "empty-state" }, "No matches.")));
      return;
    }
    filtered.forEach((c) => {
      tbody.appendChild(
        el(
          "tr",
          { onclick: () => (location.hash = `#/twin/${c.customer_id}`) },
          [
            el("td", {}, [el("div", {}, c.policy_type), el("div", { class: "mono small" }, c.customer_id)]),
            el("td", {}, c.region_name),
            el("td", {}, String(c.age)),
            el("td", {}, riskBadge(c.risk_level)),
            el("td", { class: "mono" }, pct(c.churn_probability)),
          ]
        )
      );
    });
  }

  renderRows("");
  document.getElementById("customer-search").addEventListener("input", (e) => renderRows(e.target.value));
}

// ---------------------------------------------------------------- twin / customer detail

async function renderTwin(view, customerId) {
  view.innerHTML = "";

  if (!state.customers.length) {
    state.customers = (await apiGet("/customers")).customers;
    if (state.modelAvailable === null) state.modelAvailable = state.customers.some((c) => c.risk_level !== null);
  }

  const picker = el("div", { class: "row", style: "margin-bottom:16px;" }, [
    el("div", { class: "field" }, [
      el("label", {}, "Customer"),
      buildCustomerSelect(customerId),
    ]),
  ]);
  view.appendChild(picker);

  const container = el("div", { id: "twin-detail" });
  view.appendChild(container);

  if (customerId) {
    await renderTwinDetail(container, customerId);
  } else {
    container.appendChild(el("div", { class: "card card-pad empty-state" }, "Select a customer."));
  }
}

function buildCustomerSelect(selectedId) {
  const select = el("select", {
    onchange: (e) => (location.hash = `#/twin/${e.target.value}`),
  });
  select.appendChild(el("option", { value: "" }, "Select…"));
  state.customers.forEach((c) => {
    const opt = el("option", { value: c.customer_id }, `${customerLabel(c)}`);
    if (c.customer_id === selectedId) opt.setAttribute("selected", "selected");
    select.appendChild(opt);
  });
  return select;
}

async function renderTwinDetail(container, customerId) {
  container.innerHTML = '<div class="spinner">Loading…</div>';

  const [twin, riskResult, recResult] = await Promise.allSettled([
    apiGet(`/customers/${customerId}/twin`),
    apiGet(`/customers/${customerId}/risk`),
    apiGet(`/customers/${customerId}/recommendations`),
  ]);

  container.innerHTML = "";
  if (twin.status !== "fulfilled") {
    container.appendChild(el("div", { class: "card card-pad" }, "Not found."));
    return;
  }
  const t = twin.value;
  const s = t.state;
  const hasRisk = riskResult.status === "fulfilled";

  const stabilityPanel = el("div", { class: "stability-panel", hidden: "hidden" });
  const recMcLine = el("div", { class: "rec-mc-line", hidden: "hidden" });

  const pointCol = el("div", { class: "metric-point" });
  if (hasRisk) {
    pointCol.appendChild(el("div", { class: "small muted" }, "Churn"));
    pointCol.appendChild(riskBadge(riskResult.value.risk_level));
    pointCol.appendChild(el("div", { class: "churn-num" }, pct(riskResult.value.churn_probability)));
  } else {
    pointCol.appendChild(el("div", { class: "small muted" }, "Unscored"));
  }

  const header = el("div", { class: "card card-pad", style: "margin-bottom:16px;" }, [
    el("div", { class: "twin-metrics" }, [
      el("div", {}, [
        el("div", { class: "twin-heading" }, `${s.policy_type} policy · ${s.region_name}`),
        el("div", { class: "small muted" }, `${s.age} · ${s.marital_status} · ${s.customer_tenure_months} mo · ${s.num_policies} ${s.num_policies === 1 ? "policy" : "policies"}`),
        el("div", { class: "mono small", style: "margin-top:4px;" }, `${s.customer_id} · twin v${s.version}`),
      ]),
      el("div", { class: "spread", style: "gap:20px;align-items:flex-start;" }, [pointCol, stabilityPanel]),
    ]),
  ]);
  container.appendChild(header);

  if (hasRisk) {
    stabilityPanel.hidden = false;
    stabilityPanel.appendChild(el("div", { class: "stability-title" }, "Stability range (±10% premium)"));
    stabilityPanel.appendChild(el("div", { class: "spinner" }, "Running…"));
    fetchBaselineMonteCarlo(customerId, s.current_premium)
      .then((mc) => {
        fillStabilityPanel(stabilityPanel, mc);
        recMcLine.hidden = false;
        recMcLine.textContent = recMcLineText(riskResult.value.churn_probability, mc);
      })
      .catch(() => {
        stabilityPanel.hidden = true;
        stabilityPanel.innerHTML = "";
      });
  }

  const twoCol = el("div", { class: "two-col" });

  // Twin state fields, grouped: static profile / dynamic premium & coverage /
  // dynamic payment behavior / dynamic claims / dynamic engagement & service.
  const stateCard = el("div", { class: "card card-pad" });
  stateCard.appendChild(el("div", { class: "section-title" }, "Twin state"));

  const fieldGroups = [
    {
      title: "Profile",
      fields: [
        ["Age", s.age],
        ["Region", s.region_name],
        ["Marital status", s.marital_status],
        ["Tenure (months)", s.customer_tenure_months],
        ["Multi-policy", s.multi_policy_flag ? "Yes" : "No"],
        ["Policies", s.num_policies],
        ["Payment frequency", s.payment_frequency],
        ["Autopay enabled", s.autopay_enabled ? "Yes" : "No"],
        ["Renewal month", s.renewal_month],
      ],
    },
    {
      title: "Premium & coverage",
      fields: [
        ["Current premium", "$" + s.current_premium.toFixed(2)],
        ["Premium last year", "$" + s.premium_last_year.toFixed(2)],
        ["Premium change", pct(s.premium_change_pct)],
        ["Price increases (3y)", s.num_price_increases_last_3y],
        ["Coverage", "$" + s.coverage_amount.toFixed(0)],
        ["Premium / coverage", s.premium_to_coverage_ratio.toFixed(4)],
        ["Downgraded", s.coverage_downgrade_flag ? "Yes" : "No"],
      ],
    },
    {
      title: "Payments",
      fields: [
        ["Late payments (12m)", s.late_payment_count_12m],
        ["Missed payment", s.missed_payment_flag ? "Yes" : "No"],
        ["Payment method change", s.payment_method_change_flag ? "Yes" : "No"],
      ],
    },
    {
      title: "Claims (12m)",
      fields: [
        ["Filed", s.num_claims_12m],
        ["Approved", s.num_approved_claims_12m],
        ["Rejected", s.num_rejected_claims_12m],
        ["Pending", s.num_pending_claims_12m],
        ["Avg claim", "$" + s.avg_claim_amount.toFixed(2)],
        ["Total claims", "$" + s.total_claim_amount_12m.toFixed(2)],
        ["Total payout", "$" + s.total_payout_amount_12m.toFixed(2)],
        ["Payout ratio", s.payout_ratio_12m.toFixed(3)],
        ["Settlement (days)", s.avg_settlement_time_days],
        ["Days since last", s.days_since_last_claim],
      ],
    },
    {
      title: "Engagement",
      fields: [
        ["Contacts (12m)", s.num_contacts_12m],
        ["Complaint", s.complaint_flag ? "Yes" : "No"],
        ["Resolution (days)", s.complaint_resolution_days],
        ["Quote requested", s.quote_requested_flag ? "Yes" : "No"],
      ],
    },
  ];

  fieldGroups.forEach((group) => {
    stateCard.appendChild(el("div", { class: "small muted", style: "margin:14px 0 6px 0;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;font-size:10.5px;" }, group.title));
    const grid = el("div", { class: "grid-2" });
    group.fields.forEach(([label, value]) => {
      grid.appendChild(
        el("div", {}, [el("div", { class: "small muted" }, label), el("div", { class: "mono", style: "font-weight:600;" }, String(value))])
      );
    });
    stateCard.appendChild(grid);
  });
  twoCol.appendChild(stateCard);

  // Drivers + recommendation
  const recCard = el("div", { class: "card card-pad" });
  recCard.appendChild(el("div", { class: "section-title" }, "Drivers & action"));
  if (recResult.status === "fulfilled") {
    const rec = recResult.value;
    const maxImportance = Math.max(...rec.top_drivers.map((d) => d.combined_score), 0.0001);
    rec.top_drivers.forEach((d) => {
      recCard.appendChild(
        el("div", { style: "margin-bottom:10px;" }, [
          el("div", { class: "spread small" }, [el("b", {}, d.feature), el("span", { class: "mono muted" }, String(d.value))]),
          el("div", { class: "driver-bar-track" }, el("div", { class: "driver-bar-fill", style: `width:${(d.combined_score / maxImportance) * 100}%` })),
        ])
      );
    });
    recCard.appendChild(
      el("div", { class: "callout callout-teal", style: "margin-top:12px;" }, [
        el("div", { style: "font-weight:700;margin-bottom:4px;" }, rec.recommended_action.label),
        el("div", {}, rec.recommended_action.description),
        el("div", { class: "small", style: "margin-top:6px;opacity:0.85;" }, `EV (MVP): ${rec.recommended_action.expected_value.toFixed(2)}`),
        recMcLine,
      ])
    );
  } else {
    recCard.appendChild(el("div", { class: "empty-state small" }, "No recommendations."));
  }
  twoCol.appendChild(recCard);

  container.appendChild(twoCol);

  const eventsCard = el("div", { class: "card card-pad", style: "margin-top:16px;" });
  eventsCard.appendChild(el("div", { class: "section-title" }, "Recent events"));
  const events = t.state.event_history.slice().reverse();
  if (!events.length) {
    eventsCard.appendChild(el("div", { class: "empty-state small" }, "None."));
  } else {
    const timeline = el("div", { class: "timeline" });
    events.forEach((e) => {
      timeline.appendChild(
        el("div", { class: "timeline-item" }, [
          el("div", { class: "timeline-dot" }),
          el("div", { class: "timeline-body" }, [
            el("div", {}, [el("b", {}, eventLabel(e.event_type))]),
            el("div", { class: "small muted" }, e.description || ""),
            el("div", { class: "timeline-time" }, fmtTime(e.occurred_at)),
          ]),
        ])
      );
    });
    eventsCard.appendChild(timeline);
  }
  container.appendChild(eventsCard);
}

// ---------------------------------------------------------------- simulation

const SCENARIOS = [
  {
    id: "payment_missed",
    label: "Payment missed",
    fields: [{ key: "count", label: "Count", type: "number", default: 1 }],
  },
  {
    id: "claim_created",
    label: "Claim",
    fields: [
      { key: "claim_amount", label: "Amount ($)", type: "number", default: 2000 },
      { key: "outcome", label: "Outcome", type: "select", options: ["approved", "rejected", "pending"], default: "approved" },
      { key: "settlement_time_days", label: "Settlement (days)", type: "number", default: 14 },
    ],
  },
  {
    id: "premium_changed",
    label: "Premium changed",
    fields: [{ key: "change_pct", label: "Change (e.g. 0.15)", type: "number", default: 0.15, step: "0.01" }],
  },
  { id: "policy_renewed", label: "Policy renewed", fields: [] },
  {
    id: "engagement_changed",
    label: "Engagement",
    fields: [{ key: "contact_delta", label: "Contacts", type: "number", default: 1 }],
  },
  {
    id: "coverage_downgraded",
    label: "Coverage downgrade",
    fields: [{ key: "reduction_pct", label: "Cut (e.g. 0.2)", type: "number", default: 0.2, step: "0.01" }],
  },
  {
    id: "complaint_lodged",
    label: "Complaint",
    fields: [{ key: "resolution_days", label: "Resolution (days)", type: "number", default: 7 }],
  },
];

function renderScenarioFields(container, scenarioId) {
  container.innerHTML = "";
  const scenario = SCENARIOS.find((s) => s.id === scenarioId);
  if (!scenario || !scenario.fields.length) return;
  scenario.fields.forEach((f) => {
    let input;
    if (f.type === "select") {
      input = buildPlainSelect(f.options.map((o) => [o, String(o)]), f.default);
      input.dataset.key = f.key;
    } else {
      const attrs = { type: "number", value: f.default ?? "", "data-key": f.key };
      if (f.step) attrs.step = f.step;
      input = el("input", attrs);
    }
    container.appendChild(el("div", { class: "field" }, [el("label", {}, f.label), input]));
  });
}

function collectScenarioParams(container) {
  const params = {};
  container.querySelectorAll("[data-key]").forEach((input) => {
    const key = input.dataset.key;
    const raw = input.value;
    params[key] = isNaN(Number(raw)) || raw === "" ? raw : Number(raw);
  });
  return params;
}

function sourceBadge(source) {
  const isManual = source === "manual";
  return el("span", { class: `badge ${isManual ? "badge-manual" : "badge-generator"}` }, isManual ? "Manual" : "Auto");
}

function feedItemEl(entry) {
  return el("div", { class: "timeline-item", "data-event-id": entry.event_id || "" }, [
    el("div", { class: "timeline-dot" }),
    el("div", { class: "timeline-body" }, [
      el("div", { class: "spread" }, [
        el("div", {}, [
          el("b", {}, eventLabel(entry.event_type)),
          " · ",
          el("span", { class: "mono" }, entry.customer_id || ""),
        ]),
        sourceBadge(entry.source),
      ]),
      el("div", { class: "small muted" }, entry.description || ""),
      el("div", { class: "timeline-time" }, fmtTime(entry.occurred_at)),
    ]),
  ]);
}

// ---------------------------------------------------------------- events

async function renderEvents(view) {
  view.innerHTML = "";
  if (!state.customers.length) {
    state.customers = (await apiGet("/customers")).customers;
    if (state.modelAvailable === null) state.modelAvailable = state.customers.some((c) => c.risk_level !== null);
  }
  if (currentRoute().route !== "events") return;

  const feedEntries = [];
  const seenIds = new Set();
  const sourceById = {};

  const stack = el("div", { class: "events-stack" });

  const statusCard = el("div", { class: "card card-pad" });
  statusCard.appendChild(el("div", { class: "section-title" }, "Live event source"));
  const statusBody = el("div", { id: "gen-status-body" });
  statusCard.appendChild(statusBody);

  const intervalInput = el("input", { type: "number", value: "5", min: "1", max: "60", step: "1" });
  const checksWrap = el("div", { class: "scenario-checks" });
  SCENARIOS.forEach((s) => {
    const box = el("input", { type: "checkbox", value: s.id, checked: "checked" });
    checksWrap.appendChild(el("label", { class: "scenario-check" }, [box, s.label]));
  });

  statusCard.appendChild(
    el("div", { class: "grid-2", style: "margin-top:16px;" }, [
      el("div", { class: "field" }, [el("label", {}, "Interval (s)"), intervalInput]),
      el("div", { class: "field" }, [el("label", {}, "Types"), checksWrap]),
    ])
  );

  const startBtn = el("button", { class: "btn btn-primary" }, "Start");
  const stopBtn = el("button", { class: "btn" }, "Stop");
  statusCard.appendChild(el("div", { class: "row", style: "margin-top:16px;" }, [startBtn, stopBtn]));
  stack.appendChild(statusCard);

  const fireCard = el("div", { class: "card card-pad" });
  fireCard.appendChild(el("div", { class: "section-title" }, "Fire event"));
  const customerSearch = el("input", { type: "text", placeholder: "Search…", style: "width:280px;" });
  const customerSelect = el("select");
  const eventTypeSelect = buildPlainSelect(SCENARIOS.map((s) => [s.id, s.label]), SCENARIOS[0].id);
  const payloadFields = el("div", { class: "grid-3", style: "margin-top:12px;" });

  function fillCustomerSelect(query) {
    const previous = customerSelect.value;
    customerSelect.innerHTML = "";
    const matches = state.customers.filter((c) => customerMatchesQuery(c, query));
    if (!matches.length) {
      customerSelect.appendChild(el("option", { value: "" }, "No matches"));
      return;
    }
    matches.forEach((c) => {
      const opt = el("option", { value: c.customer_id }, customerLabel(c));
      customerSelect.appendChild(opt);
    });
    if (matches.some((c) => c.customer_id === previous)) customerSelect.value = previous;
  }

  fillCustomerSelect("");
  customerSearch.addEventListener("input", () => fillCustomerSelect(customerSearch.value));
  eventTypeSelect.addEventListener("change", () => renderScenarioFields(payloadFields, eventTypeSelect.value));
  renderScenarioFields(payloadFields, eventTypeSelect.value);

  fireCard.appendChild(
    el("div", { class: "row", style: "margin-bottom:12px;" }, [
      el("div", { class: "field" }, [el("label", {}, "Search"), customerSearch]),
      el("div", { class: "field" }, [el("label", {}, "Customer"), customerSelect]),
      el("div", { class: "field" }, [el("label", {}, "Event type"), eventTypeSelect]),
    ])
  );
  fireCard.appendChild(payloadFields);
  const fireBtn = el("button", { class: "btn btn-primary" }, "Fire");
  const fireNote = el("div");
  fireCard.appendChild(el("div", { class: "row", style: "margin-top:16px;" }, [fireBtn]));
  fireCard.appendChild(fireNote);
  stack.appendChild(fireCard);

  const feedCard = el("div", { class: "card card-pad" });
  const liveIndicator = el("span", { class: "live-indicator", hidden: "hidden" }, [
    el("span", { class: "pulse" }),
    "Live",
  ]);
  feedCard.appendChild(
    el("div", { class: "spread" }, [
      el("div", { class: "section-title", style: "margin-bottom:0;" }, "Feed"),
      liveIndicator,
    ])
  );
  const feedList = el("div", { class: "timeline", style: "margin-top:16px;" });
  const feedEmpty = el("div", { class: "empty-state small" }, "No events.");
  feedCard.appendChild(feedEmpty);
  feedCard.appendChild(feedList);
  stack.appendChild(feedCard);

  view.appendChild(stack);

  function selectedScenarios() {
    return Array.from(checksWrap.querySelectorAll("input[type=checkbox]:checked")).map((box) => box.value);
  }

  function applyStatusToForm(status, syncForm) {
    if (syncForm) {
      if (status.interval_seconds != null) intervalInput.value = String(status.interval_seconds);
      if (Array.isArray(status.scenarios) && status.scenarios.length) {
        const allowed = new Set(status.scenarios);
        checksWrap.querySelectorAll("input[type=checkbox]").forEach((box) => {
          box.checked = allowed.has(box.value);
        });
      }
    }
    statusBody.innerHTML = "";
    const running = !!status.running;
    statusBody.appendChild(
      el("div", { class: "grid-3" }, [
        el("div", {}, [
          el("div", { class: "small muted" }, "Status"),
          el("div", { class: "kpi-value", style: "font-size:22px;" }, running ? "Running" : "Stopped"),
        ]),
        el("div", {}, [
          el("div", { class: "small muted" }, "Interval"),
          el("div", { class: "mono" }, `${status.interval_seconds}s`),
        ]),
        el("div", {}, [
          el("div", { class: "small muted" }, "Generated"),
          el("div", { class: "kpi-value", style: "font-size:22px;" }, String(status.events_generated ?? 0)),
        ]),
      ])
    );
    statusBody.appendChild(
      el("div", { class: "small muted", style: "margin-top:12px;" }, `Types: ${(status.scenarios || []).join(", ") || "—"}`)
    );
    startBtn.disabled = running;
    stopBtn.disabled = !running;
  }

  function setLive(on) {
    liveIndicator.hidden = !on;
  }

  function paintFeed() {
    feedList.innerHTML = "";
    if (!feedEntries.length) {
      feedEmpty.hidden = false;
      return;
    }
    feedEmpty.hidden = true;
    feedEntries.forEach((entry) => feedList.appendChild(feedItemEl(entry)));
  }

  function prependFeed(entry) {
    if (!entry) return;
    if (entry.event_id && seenIds.has(entry.event_id)) return;
    if (entry.event_id) seenIds.add(entry.event_id);
    if (entry.source) sourceById[entry.event_id] = entry.source;
    feedEntries.unshift(entry);
    if (feedEntries.length > EVENTS_FEED_CAP) {
      const dropped = feedEntries.splice(EVENTS_FEED_CAP);
      dropped.forEach((d) => {
        if (d.event_id) seenIds.delete(d.event_id);
      });
    }
    paintFeed();
  }

  function mergePolledEvents(recent) {
    const incoming = [];
    (recent || []).forEach((e) => {
      if (!e.event_id || seenIds.has(e.event_id)) return;
      incoming.push({
        event_id: e.event_id,
        customer_id: e.customer_id,
        event_type: e.event_type,
        description: e.description,
        occurred_at: e.occurred_at,
        source: sourceById[e.event_id] || "event_generator",
      });
    });
    if (!incoming.length) return;
    incoming.sort((a, b) => String(b.occurred_at).localeCompare(String(a.occurred_at)));
    incoming.reverse().forEach((entry) => prependFeed(entry));
  }

  function startPolling() {
    if (viewPollTimer != null) return;
    setLive(true);
    viewPollTimer = setInterval(async () => {
      if (currentRoute().route !== "events") {
        stopViewTimers();
        return;
      }
      try {
        const status = await apiGet("/event-generator/status");
        if (currentRoute().route !== "events") return;
        applyStatusToForm(status, false);
        refreshTopbar();
        if (!status.running) {
          stopViewTimers();
          setLive(false);
          return;
        }
        const summary = await apiGet("/dashboard/summary");
        if (currentRoute().route !== "events") return;
        mergePolledEvents(summary.recent_events);
      } catch {
        /* keep the last known feed */
      }
    }, EVENTS_POLL_MS);
  }

  startBtn.addEventListener("click", async () => {
    const scenarios = selectedScenarios();
    if (!scenarios.length) return alert("Pick at least one type.");
    try {
      const status = await apiPost("/event-generator/start", {
        interval_seconds: Number(intervalInput.value) || 5,
        scenarios,
      });
      applyStatusToForm(status, true);
      refreshTopbar();
      startPolling();
    } catch (err) {
      alert(err.detail || err.message || "Start failed.");
    }
  });

  stopBtn.addEventListener("click", async () => {
    try {
      const status = await apiPost("/event-generator/stop", {});
      applyStatusToForm(status, true);
      refreshTopbar();
      stopViewTimers();
      setLive(false);
    } catch (err) {
      alert(err.detail || err.message || "Stop failed.");
    }
  });

  fireBtn.addEventListener("click", async () => {
    const customerId = customerSelect.value;
    if (!customerId) return alert("Pick a customer.");
    fireNote.innerHTML = "";
    try {
      const result = await apiPost("/events", {
        customer_id: customerId,
        event_type: eventTypeSelect.value,
        payload: collectScenarioParams(payloadFields),
        source: "manual",
      });
      let latest = null;
      try {
        const history = await apiGet(`/customers/${customerId}/events`);
        const list = history.events || [];
        latest = [...list].reverse().find((e) => e.event_type === eventTypeSelect.value) || list[list.length - 1] || null;
      } catch {
        latest = null;
      }
      const entry = {
        event_id: latest?.event_id || `manual-${Date.now()}`,
        customer_id: customerId,
        event_type: eventTypeSelect.value,
        description: latest?.description || eventLabel(eventTypeSelect.value),
        occurred_at: latest?.occurred_at || new Date().toISOString(),
        source: "manual",
      };
      prependFeed(entry);
      fireNote.appendChild(
        el(
          "div",
          { class: "fire-confirm" },
          `${entry.description} · ${customerId} · twin v${result.twin_version}`
        )
      );
    } catch (err) {
      fireNote.appendChild(el("div", { class: "fire-error" }, err.detail || err.message || "Fire failed."));
    }
  });

  try {
    const [status, summary] = await Promise.all([apiGet("/event-generator/status"), apiGet("/dashboard/summary")]);
    applyStatusToForm(status, true);
    mergePolledEvents(summary.recent_events);
    if (currentRoute().route !== "events") return;
    if (status.running) startPolling();
    else setLive(false);
  } catch {
    applyStatusToForm({ running: false, interval_seconds: 5, scenarios: SCENARIOS.map((s) => s.id), events_generated: 0 }, true);
  }
}

async function renderSimulation(view, customerIdParam) {
  view.innerHTML = "";
  if (!state.customers.length) {
    state.customers = (await apiGet("/customers")).customers;
    if (state.modelAvailable === null) state.modelAvailable = state.customers.some((c) => c.risk_level !== null);
  }

  const form = el("div", { class: "card card-pad" });
  const customerSelect = buildPlainSelect(
    state.customers.map((c) => [c.customer_id, customerLabel(c)]),
    customerIdParam || state.selectedCustomerId
  );
  const scenarioSelect = buildPlainSelect(SCENARIOS.map((s) => [s.id, s.label]), SCENARIOS[0].id);
  const paramsContainer = el("div", { class: "grid-3", style: "margin-top:12px;" });
  const trialsInput = el("input", { type: "number", value: "300", min: "10", max: "5000" });
  const noiseInput = el("input", { type: "number", value: "0.10", step: "0.01", min: "0", max: "1" });

  scenarioSelect.addEventListener("change", () => renderScenarioFields(paramsContainer, scenarioSelect.value));
  renderScenarioFields(paramsContainer, scenarioSelect.value);

  form.appendChild(
    el("div", { class: "grid-3" }, [
      el("div", { class: "field" }, [el("label", {}, "Customer"), customerSelect]),
      el("div", { class: "field" }, [el("label", {}, "Scenario"), scenarioSelect]),
      el("div"),
    ])
  );
  form.appendChild(paramsContainer);

  const runRow = el("div", { class: "row", style: "margin-top:16px;" });
  const runBtn = el("button", { class: "btn btn-primary" }, "Run simulation");
  runRow.appendChild(runBtn);
  form.appendChild(runRow);

  const advanced = el("details", { class: "advanced-block" });
  advanced.appendChild(el("summary", {}, "Advanced"));
  advanced.appendChild(
    el("div", { class: "advanced-fields" }, [
      el("div", { class: "field" }, [el("label", {}, "MC trials"), trialsInput]),
      el("div", { class: "field" }, [el("label", {}, "Noise (std)"), noiseInput]),
    ])
  );
  form.appendChild(advanced);

  view.appendChild(form);

  const resultsContainer = el("div", { id: "sim-results", style: "margin-top:16px;" });
  view.appendChild(resultsContainer);

  runBtn.addEventListener("click", async () => {
    const customerId = customerSelect.value;
    if (!customerId) return alert("Pick a customer.");
    const scenario = scenarioSelect.value;
    const parameters = collectScenarioParams(paramsContainer);
    const trials = Number(trialsInput.value) || 300;
    const numeric_noise_std = Number(noiseInput.value) || 0.1;

    resultsContainer.innerHTML = '<div class="spinner">Running…</div>';

    const detPromise = apiPost(`/customers/${customerId}/simulate`, { scenario, parameters });
    const mcPromise = apiPost(`/customers/${customerId}/simulate/monte-carlo`, {
      scenario,
      parameters,
      trials,
      numeric_noise_std,
    });

    let detResult;
    try {
      detResult = await detPromise;
    } catch (err) {
      renderSimError(resultsContainer, err);
      return;
    }

    renderCombinedSimulationResult(resultsContainer, detResult, null, true);

    try {
      const mcResult = await mcPromise;
      renderCombinedSimulationResult(resultsContainer, detResult, mcResult, false);
    } catch (err) {
      renderCombinedSimulationResult(resultsContainer, detResult, null, false, err);
    }
  });
}

function buildPlainSelect(pairs, selectedValue) {
  const select = el("select");
  pairs.forEach(([value, label]) => {
    const opt = el("option", { value: String(value) }, label);
    if (String(value) === String(selectedValue)) opt.setAttribute("selected", "selected");
    select.appendChild(opt);
  });
  return select;
}

function renderSimError(container, err) {
  container.innerHTML = "";
  container.appendChild(
    el("div", { class: "card card-pad callout-warn" }, err.detail || err.message || "Simulation failed.")
  );
}

function renderCombinedSimulationResult(container, detResult, mcResult, mcLoading, mcError) {
  container.innerHTML = "";
  const diff = detResult.difference;
  const diffClass = diff > 0 ? "diff-up" : diff < 0 ? "diff-down" : "";
  const card = el("div", { class: "card card-pad" });
  card.appendChild(el("div", { class: "section-title" }, "Result"));
  card.appendChild(
    el("div", { class: "grid-3" }, [
      el("div", {}, [el("div", { class: "small muted" }, "Before"), el("div", { class: "kpi-value", style: "font-size:28px;" }, pct(detResult.before.churn_probability)), riskBadge(detResult.before.risk_level)]),
      el("div", {}, [el("div", { class: "small muted" }, "After"), el("div", { class: "kpi-value", style: "font-size:28px;" }, pct(detResult.after.churn_probability)), riskBadge(detResult.after.risk_level)]),
      el("div", {}, [el("div", { class: "small muted" }, "Difference"), el("div", { class: `kpi-value ${diffClass}`, style: "font-size:28px;" }, (diff > 0 ? "+" : "") + pct(diff))]),
    ])
  );
  card.appendChild(
    el("div", { class: "callout callout-teal", style: "margin-top:14px;" }, "Simulation only — real Twin unchanged.")
  );

  const mcSection = el("div", { style: "margin-top:24px;" });
  mcSection.appendChild(el("div", { class: "section-title" }, "Monte Carlo"));
  if (mcLoading) {
    mcSection.appendChild(el("div", { class: "spinner" }, "Running…"));
  } else if (mcError) {
    mcSection.appendChild(el("div", { class: "sim-mc-error" }, mcError.detail || mcError.message || "Monte Carlo failed."));
  } else if (mcResult) {
    mcSection.appendChild(
      el("div", { class: "grid-3" }, [
        statBox("Mean", pct(mcResult.mean_churn_probability)),
        statBox("Median", pct(mcResult.median_churn_probability)),
        statBox("Std dev", pct(mcResult.std_dev)),
        statBox("P10", pct(mcResult.p10_churn_probability)),
        statBox("P90", pct(mcResult.p90_churn_probability)),
        statBox("Trials", String(mcResult.trials)),
      ])
    );
    mcSection.appendChild(el("div", { class: "small muted", style: "margin-top:14px;" }, "Churn probability (low → high)"));
    mcSection.appendChild(histogramEl(mcResult.distribution_sample));
    if (mcResult.assumptions) {
      mcSection.appendChild(
        el("div", { class: "callout callout-teal", style: "margin-top:14px;" }, "MVP: ±10% parameter noise, not model uncertainty.")
      );
    }
  }
  card.appendChild(mcSection);
  container.appendChild(card);
}

function statBox(label, value) {
  return el("div", {}, [el("div", { class: "small muted" }, label), el("div", { class: "kpi-value", style: "font-size:20px;" }, value)]);
}
