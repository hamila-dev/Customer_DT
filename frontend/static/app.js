// Customer Twin — simple vanilla-JS single-page app.
// No build step, no framework: fetch() against the FastAPI backend under
// /api/*, hash-based routing, and small render() functions per view.

const API = "/api";

const state = {
  customers: [],
  selectedCustomerId: null,
  modelAvailable: null, // null = unknown yet, true/false once known
};

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

// ---------------------------------------------------------------- topbar

async function refreshTopbar() {
  try {
    const status = await apiGet("/event-generator/status");
    const dot = document.getElementById("gen-dot");
    const label = document.getElementById("gen-label");
    dot.className = "dot " + (status.running ? "dot-on" : "dot-off");
    label.textContent = status.running
      ? `Event generator: running (${status.events_generated} events)`
      : "Event generator: stopped";
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
      pill.textContent = state.modelAvailable ? "model loaded" : "model not loaded";
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
  simulation: { title: "Simulation", render: renderSimulation },
};

function currentRoute() {
  const hash = location.hash.replace(/^#\//, "");
  const [route, param] = hash.split("/");
  return { route: routes[route] ? route : "dashboard", param };
}

async function router() {
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
        el("div", { class: "section-title" }, "Something went wrong"),
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
    kpiCard("Total customers", summary.total_customers, ""),
    kpiCard("High risk", summary.high_risk, "high"),
    kpiCard("Medium risk", summary.medium_risk, "medium"),
    kpiCard("Low risk", summary.low_risk, "low"),
  ]);
  view.appendChild(kpis);

  if (!summary.model_available) {
    view.appendChild(
      el("div", { class: "callout callout-warn", style: "margin-bottom:16px;" }, [
        el("b", {}, "No trained model loaded. "),
        "Risk scores, drivers, and recommendations are unavailable until model/churn_model.joblib and model/preprocessing.joblib are added. See model/README.md.",
      ])
    );
  }

  const twoCol = el("div", { class: "two-col" });

  // High-risk customer list
  const highRiskCard = el("div", { class: "card card-pad" });
  highRiskCard.appendChild(el("div", { class: "section-title" }, "High-risk customers"));
  if (!summary.high_risk_customers.length) {
    highRiskCard.appendChild(el("div", { class: "empty-state small" }, "No high-risk customers right now."));
  } else {
    const table = el("table", {}, [
      el("thead", {}, el("tr", {}, [el("th", {}, "Customer"), el("th", {}, "Churn probability")])),
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
    eventsCard.appendChild(el("div", { class: "empty-state small" }, "No events yet. Start the event generator from the Digital Twin view."));
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
    el("input", { type: "text", placeholder: "Search customers…", id: "customer-search", style: "width:280px;" }),
    el("span", { class: "small muted" }, `${data.total} customers loaded from the prototype dataset`),
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
        el("th", {}, "Churn probability"),
      ])
    ),
  ]);
  const tbody = el("tbody");
  table.appendChild(tbody);
  card.appendChild(table);
  view.appendChild(card);

  function renderRows(filterText) {
    tbody.innerHTML = "";
    const filtered = data.customers.filter((c) =>
      !filterText ||
      customerLabel(c).toLowerCase().includes(filterText.toLowerCase()) ||
      c.customer_id.includes(filterText)
    );
    if (!filtered.length) {
      tbody.appendChild(el("tr", {}, el("td", { colspan: "5", class: "empty-state" }, "No customers match your search.")));
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
    container.appendChild(el("div", { class: "card card-pad empty-state" }, "Select a customer to view their Digital Twin."));
  }
}

function buildCustomerSelect(selectedId) {
  const select = el("select", {
    onchange: (e) => (location.hash = `#/twin/${e.target.value}`),
  });
  select.appendChild(el("option", { value: "" }, "Select a customer…"));
  state.customers.forEach((c) => {
    const opt = el("option", { value: c.customer_id }, `${customerLabel(c)}`);
    if (c.customer_id === selectedId) opt.setAttribute("selected", "selected");
    select.appendChild(opt);
  });
  return select;
}

async function renderTwinDetail(container, customerId) {
  container.innerHTML = '<div class="spinner">Loading customer twin…</div>';

  const [twin, riskResult, recResult] = await Promise.allSettled([
    apiGet(`/customers/${customerId}/twin`),
    apiGet(`/customers/${customerId}/risk`),
    apiGet(`/customers/${customerId}/recommendations`),
  ]);

  container.innerHTML = "";
  if (twin.status !== "fulfilled") {
    container.appendChild(el("div", { class: "card card-pad" }, "Customer not found."));
    return;
  }
  const t = twin.value;
  const s = t.state;

  const header = el("div", { class: "card card-pad", style: "margin-bottom:16px;" }, [
    el("div", { class: "spread" }, [
      el("div", {}, [
        el("div", { style: "font-size:16px;font-weight:700;color:var(--navy);" }, `${s.policy_type} policy · ${s.region_name}`),
        el("div", { class: "small muted" }, `Age ${s.age} · ${s.marital_status} · ${s.customer_tenure_months} months tenure · ${s.num_policies} polic${s.num_policies === 1 ? "y" : "ies"}`),
        el("div", { class: "mono small", style: "margin-top:4px;" }, `${s.customer_id} · twin v${s.version}`),
      ]),
      el(
        "div",
        { style: "text-align:right;" },
        riskResult.status === "fulfilled"
          ? [riskBadge(riskResult.value.risk_level), el("div", { class: "mono", style: "font-size:20px;margin-top:6px;" }, pct(riskResult.value.churn_probability))]
          : el("div", { class: "small muted" }, "Risk score unavailable (model not loaded)")
      ),
    ]),
  ]);
  container.appendChild(header);

  const twoCol = el("div", { class: "two-col" });

  // Twin state fields, grouped: static profile / dynamic premium & coverage /
  // dynamic payment behavior / dynamic claims / dynamic engagement & service.
  const stateCard = el("div", { class: "card card-pad" });
  stateCard.appendChild(el("div", { class: "section-title" }, "Twin state"));

  const fieldGroups = [
    {
      title: "Profile (static)",
      fields: [
        ["Age", s.age],
        ["Region", s.region_name],
        ["Marital status", s.marital_status],
        ["Tenure (months)", s.customer_tenure_months],
        ["Multi-policy", s.multi_policy_flag ? "Yes" : "No"],
        ["Num policies", s.num_policies],
        ["Payment frequency", s.payment_frequency],
        ["Autopay enabled", s.autopay_enabled ? "Yes" : "No"],
        ["Renewal month", s.renewal_month],
      ],
    },
    {
      title: "Premium & coverage (dynamic)",
      fields: [
        ["Current premium", "$" + s.current_premium.toFixed(2)],
        ["Premium last year", "$" + s.premium_last_year.toFixed(2)],
        ["Premium change %", pct(s.premium_change_pct)],
        ["Price increases (3y)", s.num_price_increases_last_3y],
        ["Coverage amount", "$" + s.coverage_amount.toFixed(0)],
        ["Premium/coverage ratio", s.premium_to_coverage_ratio.toFixed(4)],
        ["Coverage downgraded", s.coverage_downgrade_flag ? "Yes" : "No"],
      ],
    },
    {
      title: "Payment behavior (dynamic)",
      fields: [
        ["Late payments (12m)", s.late_payment_count_12m],
        ["Missed payment flag", s.missed_payment_flag ? "Yes" : "No"],
        ["Payment method changed", s.payment_method_change_flag ? "Yes" : "No"],
      ],
    },
    {
      title: "Claims (last 12 months, dynamic)",
      fields: [
        ["Claims filed", s.num_claims_12m],
        ["Approved", s.num_approved_claims_12m],
        ["Rejected", s.num_rejected_claims_12m],
        ["Pending", s.num_pending_claims_12m],
        ["Avg claim amount", "$" + s.avg_claim_amount.toFixed(2)],
        ["Total claim amount", "$" + s.total_claim_amount_12m.toFixed(2)],
        ["Total payout amount", "$" + s.total_payout_amount_12m.toFixed(2)],
        ["Payout ratio", s.payout_ratio_12m.toFixed(3)],
        ["Avg settlement (days)", s.avg_settlement_time_days],
        ["Days since last claim", s.days_since_last_claim],
      ],
    },
    {
      title: "Engagement & service (dynamic)",
      fields: [
        ["Contacts (12m)", s.num_contacts_12m],
        ["Complaint lodged", s.complaint_flag ? "Yes" : "No"],
        ["Complaint resolution (days)", s.complaint_resolution_days],
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
  recCard.appendChild(el("div", { class: "section-title" }, "Risk drivers & recommendation"));
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
        el("div", { class: "small", style: "margin-top:6px;opacity:0.85;" }, `Expected value (assumption-based): ${rec.recommended_action.expected_value.toFixed(2)}`),
      ])
    );
  } else {
    recCard.appendChild(el("div", { class: "empty-state small" }, "Recommendations unavailable (model not loaded)."));
  }
  twoCol.appendChild(recCard);

  container.appendChild(twoCol);

  // Event timeline + generator controls
  const eventsCard = el("div", { class: "card card-pad", style: "margin-top:16px;" });
  eventsCard.appendChild(
    el("div", { class: "spread" }, [
      el("div", { class: "section-title" }, "Recent events"),
      genControls(),
    ])
  );
  const events = t.state.event_history.slice().reverse();
  if (!events.length) {
    eventsCard.appendChild(el("div", { class: "empty-state small" }, "No events yet for this customer."));
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

function genControls() {
  const startBtn = el("button", { class: "btn btn-primary" }, "Start generator");
  const stopBtn = el("button", { class: "btn" }, "Stop generator");
  startBtn.addEventListener("click", async () => {
    await apiPost("/event-generator/start", {});
    router();
  });
  stopBtn.addEventListener("click", async () => {
    await apiPost("/event-generator/stop", {});
    router();
  });
  return el("div", { class: "row" }, [startBtn, stopBtn]);
}

// ---------------------------------------------------------------- simulation

const SCENARIOS = [
  {
    id: "payment_missed",
    label: "Payment missed",
    fields: [{ key: "count", label: "Number of missed payments", type: "number", default: 1 }],
  },
  {
    id: "claim_created",
    label: "New claim filed",
    fields: [
      { key: "claim_amount", label: "Claim amount ($)", type: "number", default: 2000 },
      { key: "outcome", label: "Outcome", type: "select", options: ["approved", "rejected", "pending"], default: "approved" },
      { key: "settlement_time_days", label: "Settlement time (days)", type: "number", default: 14 },
    ],
  },
  {
    id: "premium_changed",
    label: "Premium changed",
    fields: [{ key: "change_pct", label: "Change (%, e.g. 0.15 = +15%)", type: "number", default: 0.15, step: "0.01" }],
  },
  { id: "policy_renewed", label: "Policy renewed (resets 12m counters)", fields: [] },
  {
    id: "engagement_changed",
    label: "Customer engagement changed",
    fields: [{ key: "contact_delta", label: "Additional contacts", type: "number", default: 1 }],
  },
  {
    id: "coverage_downgraded",
    label: "Coverage downgraded",
    fields: [{ key: "reduction_pct", label: "Reduction (%, e.g. 0.2 = -20%)", type: "number", default: 0.2, step: "0.01" }],
  },
  {
    id: "complaint_lodged",
    label: "Complaint lodged",
    fields: [{ key: "resolution_days", label: "Resolution time (days)", type: "number", default: 7 }],
  },
];

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

  function renderParamFields() {
    paramsContainer.innerHTML = "";
    const scenario = SCENARIOS.find((s) => s.id === scenarioSelect.value);
    scenario.fields.forEach((f) => {
      let input;
      if (f.type === "select") {
        input = buildPlainSelect(f.options.map((o) => [o, String(o)]), f.default);
        input.dataset.key = f.key;
      } else {
        input = el("input", { type: "number", value: f.default ?? "", "data-key": f.key });
      }
      paramsContainer.appendChild(el("div", { class: "field" }, [el("label", {}, f.label), input]));
    });
  }
  scenarioSelect.addEventListener("change", renderParamFields);
  renderParamFields();

  form.appendChild(
    el("div", { class: "grid-3" }, [
      el("div", { class: "field" }, [el("label", {}, "Customer"), customerSelect]),
      el("div", { class: "field" }, [el("label", {}, "Scenario"), scenarioSelect]),
      el("div"),
    ])
  );
  form.appendChild(paramsContainer);

  const runRow = el("div", { class: "row", style: "margin-top:16px;" });
  const detBtn = el("button", { class: "btn btn-primary" }, "Run deterministic simulation");
  const mcBtn = el("button", { class: "btn" }, "Run Monte Carlo simulation");
  runRow.appendChild(detBtn);
  runRow.appendChild(mcBtn);
  runRow.appendChild(el("div", { class: "field" }, [el("label", {}, "MC trials"), trialsInput]));
  runRow.appendChild(el("div", { class: "field" }, [el("label", {}, "Numeric noise (std)"), noiseInput]));
  form.appendChild(runRow);

  view.appendChild(form);

  const resultsContainer = el("div", { id: "sim-results", style: "margin-top:16px;" });
  view.appendChild(resultsContainer);

  function collectParams() {
    const params = {};
    paramsContainer.querySelectorAll("[data-key]").forEach((input) => {
      const key = input.dataset.key;
      const raw = input.value;
      params[key] = isNaN(Number(raw)) || raw === "" ? raw : Number(raw);
    });
    return params;
  }

  detBtn.addEventListener("click", async () => {
    const customerId = customerSelect.value;
    if (!customerId) return alert("Select a customer first.");
    resultsContainer.innerHTML = '<div class="spinner">Running deterministic simulation…</div>';
    try {
      const result = await apiPost(`/customers/${customerId}/simulate`, {
        scenario: scenarioSelect.value,
        parameters: collectParams(),
      });
      renderDeterministicResult(resultsContainer, result);
    } catch (err) {
      renderSimError(resultsContainer, err);
    }
  });

  mcBtn.addEventListener("click", async () => {
    const customerId = customerSelect.value;
    if (!customerId) return alert("Select a customer first.");
    resultsContainer.innerHTML = '<div class="spinner">Running Monte Carlo simulation…</div>';
    try {
      const result = await apiPost(`/customers/${customerId}/simulate/monte-carlo`, {
        scenario: scenarioSelect.value,
        parameters: collectParams(),
        trials: Number(trialsInput.value) || 300,
        numeric_noise_std: Number(noiseInput.value) || 0.1,
      });
      renderMonteCarloResult(resultsContainer, result);
    } catch (err) {
      renderSimError(resultsContainer, err);
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

function renderDeterministicResult(container, result) {
  container.innerHTML = "";
  const diff = result.difference;
  const diffClass = diff > 0 ? "diff-up" : diff < 0 ? "diff-down" : "";
  const card = el("div", { class: "card card-pad" });
  card.appendChild(el("div", { class: "section-title" }, "Deterministic what-if result"));
  card.appendChild(
    el("div", { class: "grid-3" }, [
      el("div", {}, [el("div", { class: "small muted" }, "Before"), el("div", { class: "mono", style: "font-size:22px;" }, pct(result.before.churn_probability)), riskBadge(result.before.risk_level)]),
      el("div", {}, [el("div", { class: "small muted" }, "After"), el("div", { class: "mono", style: "font-size:22px;" }, pct(result.after.churn_probability)), riskBadge(result.after.risk_level)]),
      el("div", {}, [el("div", { class: "small muted" }, "Difference"), el("div", { class: `mono ${diffClass}`, style: "font-size:22px;" }, (diff > 0 ? "+" : "") + pct(diff))]),
    ])
  );
  card.appendChild(
    el("div", { class: "callout callout-teal", style: "margin-top:14px;" }, "The real Twin state was not modified by this simulation — only a cloned copy was scored.")
  );
  container.appendChild(card);
}

function renderMonteCarloResult(container, result) {
  container.innerHTML = "";
  const card = el("div", { class: "card card-pad" });
  card.appendChild(el("div", { class: "section-title" }, `Monte Carlo result (${result.trials} trials)`));
  card.appendChild(
    el("div", { class: "grid-3" }, [
      statBox("Mean", pct(result.mean_churn_probability)),
      statBox("Median", pct(result.median_churn_probability)),
      statBox("Std dev", pct(result.std_dev)),
      statBox("P10", pct(result.p10_churn_probability)),
      statBox("P90", pct(result.p90_churn_probability)),
      statBox("Trials", String(result.trials)),
    ])
  );

  // Simple histogram from the distribution sample.
  const buckets = new Array(20).fill(0);
  result.distribution_sample.forEach((v) => {
    const idx = Math.min(19, Math.max(0, Math.floor(v * 20)));
    buckets[idx] += 1;
  });
  const maxBucket = Math.max(...buckets, 1);
  const hist = el(
    "div",
    { class: "hist-bars", style: "margin-top:16px;" },
    buckets.map((count) => el("div", { class: "hist-bar", style: `height:${(count / maxBucket) * 100}%` }))
  );
  card.appendChild(el("div", { class: "small muted", style: "margin-top:14px;" }, "Outcome distribution (churn probability, low → high)"));
  card.appendChild(hist);

  card.appendChild(
    el("div", { class: "callout callout-teal", style: "margin-top:14px;" }, [
      el("div", {}, result.assumptions.source_of_stochasticity),
      el("div", { class: "small", style: "margin-top:6px;opacity:0.85;" }, result.assumptions.note),
    ])
  );
  container.appendChild(card);
}

function statBox(label, value) {
  return el("div", {}, [el("div", { class: "small muted" }, label), el("div", { class: "mono", style: "font-size:18px;" }, value)]);
}
