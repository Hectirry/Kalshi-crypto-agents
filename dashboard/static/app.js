const statusLine = document.querySelector(".status-line");
const connectionStatus = document.getElementById("connection-status");
const refreshButton = document.getElementById("refresh-now");
const marketCount = document.getElementById("market-count");
const openCount = document.getElementById("open-count");
const closedCount = document.getElementById("closed-count");
const totalPnl = document.getElementById("total-pnl");
const realizedPnl = document.getElementById("realized-pnl");
const openPnl = document.getElementById("open-pnl");
const blockedCount = document.getElementById("blocked-count");
const lastRefresh = document.getElementById("last-refresh");
const effectiveMode = document.getElementById("effective-mode");
const decisionSummary = document.getElementById("decision-summary");
const skipSummary = document.getElementById("skip-summary");
const blockedList = document.getElementById("blocked-list");
const marketCards = document.getElementById("market-cards");
const signalsList = document.getElementById("signals-list");
const openTradesList = document.getElementById("open-trades-list");
const closedTradesList = document.getElementById("closed-trades-list");
const manualFeedback = document.getElementById("manual-feedback");
const manualActionHint = document.getElementById("manual-action-hint");

const REFRESH_MS = 5000;

function formatProbability(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMoney(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return Number(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

function formatSignedMoney(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  const amount = Number(value);
  return `${amount > 0 ? "+" : ""}${formatMoney(amount)}`;
}

function formatNumber(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return Number(value).toLocaleString("en-US");
}

function formatDuration(seconds) {
  const safeSeconds = Math.max(0, Number(seconds || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatTime(timestamp) {
  if (!timestamp) {
    return "--:--:--";
  }
  return new Date(timestamp * 1000).toLocaleTimeString();
}

function pnlClass(value) {
  const amount = Number(value || 0);
  if (amount > 0) {
    return "positive";
  }
  if (amount < 0) {
    return "negative";
  }
  return "flat";
}

function decisionClass(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9_-]/g, "");
}

function slugify(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "market";
}

function marketUrl(value, ticker, eventTicker = null, title = null) {
  if (value && value !== "undefined") {
    return value;
  }
  const cleanTicker = String(ticker || "").trim();
  if (!cleanTicker) {
    return "https://kalshi.com/markets";
  }
  const parts = cleanTicker.split("-").filter(Boolean);
  const effectiveEventTicker = String(eventTicker || (parts.length > 1 ? parts.slice(0, -1).join("-") : cleanTicker)).toLowerCase();
  const category = cleanTicker.includes("BTC") ? "btc" : cleanTicker.includes("ETH") ? "eth" : cleanTicker.includes("SOL") ? "sol" : "market";
  const effectiveTitle = title || (cleanTicker.includes("15M")
    ? `${category}-price-up-in-next-15-mins`
    : `${category}-market`);
  return `https://kalshi.com/markets/${encodeURIComponent(effectiveEventTicker)}/${encodeURIComponent(slugify(effectiveTitle))}/${encodeURIComponent(cleanTicker.toLowerCase())}`;
}

function explainReason(raw) {
  const reason = String(raw || "").trim();
  if (!reason) {
    return "Sin detalle reciente";
  }
  const known = {
    blocked_category: "La categoría está bloqueada por performance reciente.",
    delta_too_small: "El edge frente al mercado todavía es demasiado pequeño.",
    ev_negative: "La operación no supera el umbral de EV neto.",
    contract_price_out_of_range: "El precio del contrato quedó fuera del rango permitido.",
    missing_or_stale_price: "Falta precio spot fresco para validar el mercado.",
    cross_exchange_divergence: "Binance y Hyperliquid no coinciden lo suficiente.",
    low_time_remaining: "Queda poco tiempo antes del vencimiento.",
    timing_not_allowed: "El filtro temporal frenó la entrada.",
  };
  if (known[reason]) {
    return known[reason];
  }
  if (reason.startsWith("manual_dashboard:")) {
    return `Acción manual: ${reason.replace("manual_dashboard:", "")}`;
  }
  if (reason.startsWith("undervalued_yes")) {
    return "El modelo vio valor en YES y generó entrada.";
  }
  if (reason.startsWith("undervalued_no")) {
    return "El modelo vio valor en NO y generó entrada.";
  }
  return reason.replaceAll("_", " ");
}

function setStatus(kind, text) {
  statusLine.classList.remove("ok", "bad");
  if (kind) {
    statusLine.classList.add(kind);
  }
  connectionStatus.textContent = text;
}

async function fetchJson(path, options = undefined) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `${path} ${response.status}`);
  }
  return payload;
}

function renderMetricPnl(element, value) {
  element.textContent = formatSignedMoney(value);
  element.classList.remove("positive", "negative", "flat");
  element.classList.add(pnlClass(value));
}

function renderDecisionSummary(decisions) {
  const entries = Object.entries(decisions || {});
  if (!entries.length) {
    decisionSummary.innerHTML = '<p class="empty small">No hay señales recientes.</p>';
    return;
  }
  decisionSummary.innerHTML = entries
    .sort((left, right) => right[1] - left[1])
    .map(([decision, count]) => `
      <div class="summary-chip">
        <span class="badge ${decisionClass(decision)}">${decision}</span>
        <strong>${count}</strong>
      </div>
    `)
    .join("");
}

function renderSkipSummary(items) {
  if (!items.length) {
    skipSummary.innerHTML = '<p class="empty small">Sin razones de bloqueo recientes.</p>';
    return;
  }
  skipSummary.innerHTML = items.map((item) => `
    <div class="reason-row">
      <span>${item.reason}</span>
      <strong>${item.count}</strong>
    </div>
  `).join("");
}

function renderBlockedList(items) {
  if (!items.length) {
    blockedList.innerHTML = '<p class="empty small">No hay categorías bloqueadas.</p>';
    return;
  }
  blockedList.innerHTML = items.map((item) => `
    <div class="summary-chip">
      <span class="badge blocked">${item}</span>
    </div>
  `).join("");
}

function estimatedOpenTradePnl(trade, marketByTicker) {
  const market = marketByTicker.get(trade.ticker);
  if (!market) {
    return { currentPrice: null, pnl: null };
  }
  const currentPrice = trade.side === "YES"
    ? Number(market.implied_prob)
    : Math.max(0.01, Math.min(0.99, 1 - Number(market.implied_prob)));
  const pnl = (currentPrice - Number(trade.entry_price)) * Number(trade.contracts);
  return { currentPrice, pnl };
}

function renderMarkets(markets, state) {
  if (!markets.length) {
    marketCards.innerHTML = '<p class="empty">Sin mercados activos ahora.</p>';
    return;
  }

  const blocked = new Set(state.blocked_categories || []);
  const manualEnabled = Boolean(state.manual_actions_enabled ?? (state.effective_mode === "demo"));
  marketCards.innerHTML = markets.map((market) => {
    const marketBlocked = blocked.has(market.category);
    const flow = market.flow || {};
    const checks = Array.isArray(flow.checks) ? flow.checks : [];
    const latestSignal = flow.latest_signal || null;
    return `
      <article class="market-card">
        <div class="market-header">
          <div>
            <p class="market-overline">${market.category}</p>
            <h3>${market.label}</h3>
            <p class="market-subtitle">${market.subtitle}</p>
          </div>
          <span class="pill ${String(market.category || "").toLowerCase()}">${market.category}</span>
        </div>
        <div class="market-grid">
          <div><span>YES ask</span><strong>${formatProbability(market.yes_ask)}</strong></div>
          <div><span>NO ask</span><strong>${formatProbability(market.no_ask)}</strong></div>
          <div><span>Volumen 24h</span><strong>${formatNumber(market.volume_24h)}</strong></div>
          <div><span>Expira</span><strong>${formatDuration(market.time_to_expiry_s)}</strong></div>
        </div>
        <div class="market-meta-row">
          <span class="market-flag ${decisionClass(flow.status || (marketBlocked ? "blocked" : "ready"))}">
            ${flow.summary || (marketBlocked ? "Bloqueado por categoría" : "Disponible para evaluación")}
          </span>
          <a class="ghost-link" href="${marketUrl(market.market_url, market.ticker, market.event_ticker, market.title)}" target="_blank" rel="noreferrer">Abrir mercado</a>
        </div>
        <div class="market-flow">
          ${(latestSignal || checks.length) ? `
            <div class="flow-explainer">
              <strong>Última lectura del motor</strong>
              <p>${latestSignal ? explainReason(latestSignal.reasoning || latestSignal.decision) : "Todavía no hay señal reciente para este ticker."}</p>
            </div>
          ` : ""}
          ${checks.length ? `
            <div class="check-grid">
              ${checks.map((check) => `
                <div class="check-chip ${decisionClass(check.status)}">
                  <span>${check.label}</span>
                  <strong>${check.status === "pass" ? "OK" : check.status === "warn" ? "YA OPERADO" : "FALTA"}</strong>
                  <small>${check.detail}</small>
                </div>
              `).join("")}
            </div>
          ` : ""}
        </div>
        <div class="market-actions">
          <button type="button" class="action-button yes" data-manual-side="YES" data-ticker="${market.ticker}" ${!manualEnabled ? "disabled" : ""}>Paper YES</button>
          <button type="button" class="action-button no" data-manual-side="NO" data-ticker="${market.ticker}" ${!manualEnabled ? "disabled" : ""}>Paper NO</button>
        </div>
      </article>
    `;
  }).join("");
}

function renderSignals(signals) {
  if (!signals.length) {
    signalsList.innerHTML = '<p class="empty">Sin señales todavía.</p>';
    return;
  }

  signalsList.innerHTML = signals.slice().reverse().map((signal) => `
    <article class="list-item">
      <div class="list-top">
        <div>
          <span class="list-title">${signal.label}</span>
          <a class="inline-link" href="${marketUrl(signal.market_url, signal.ticker)}" target="_blank" rel="noreferrer">Ver mercado</a>
        </div>
        <span class="badge ${decisionClass(signal.decision)}">${signal.decision}</span>
      </div>
      <div class="meta">
        <span>Delta ${formatProbability(signal.delta)}</span>
        <span>EV ${formatProbability(signal.ev_net_fees)}</span>
        <span>${signal.confidence}</span>
        <span>${formatDuration(signal.time_remaining_s)}</span>
      </div>
      <p class="reasoning">${explainReason(signal.reasoning)}</p>
    </article>
  `).join("");
}

function renderOpenTrades(trades, marketByTicker, effectiveModeValue) {
  if (!trades.length) {
    openTradesList.innerHTML = '<p class="empty">Sin posiciones abiertas. Revisa arriba las razones de skip y los mercados disponibles.</p>';
    return 0;
  }

  let aggregatePnl = 0;
  openTradesList.innerHTML = trades.map((trade) => {
    const estimate = estimatedOpenTradePnl(trade, marketByTicker);
    if (estimate.pnl !== null) {
      aggregatePnl += estimate.pnl;
    }
    return `
      <article class="list-item trade-card">
        <div class="list-top">
          <div>
            <span class="list-title">${trade.label}</span>
            <a class="inline-link" href="${marketUrl(trade.market_url, trade.ticker)}" target="_blank" rel="noreferrer">Abrir mercado</a>
          </div>
          <span class="badge ${decisionClass(trade.side)}">${trade.side}</span>
        </div>
        <div class="meta">
          <span>${trade.contracts} contratos</span>
          <span>Entrada ${formatProbability(trade.entry_price)}</span>
          <span>Mark ${formatProbability(estimate.currentPrice)}</span>
          <span>${trade.mode}</span>
        </div>
        <div class="trade-summary">
          <span class="trade-pill">Abierto</span>
          <span class="trade-pnl ${pnlClass(estimate.pnl)}">${formatSignedMoney(estimate.pnl)}</span>
        </div>
        <div class="trade-actions">
          <button type="button" class="secondary-button" data-close-trade="${trade.id}" ${effectiveModeValue !== "demo" ? "disabled" : ""}>Cerrar en paper</button>
        </div>
      </article>
    `;
  }).join("");

  return aggregatePnl;
}

function renderClosedTrades(trades) {
  if (!trades.length) {
    closedTradesList.innerHTML = '<p class="empty">Sin trades cerrados.</p>';
    return;
  }

  closedTradesList.innerHTML = trades.slice().reverse().map((trade) => `
    <article class="list-item trade-card">
      <div class="list-top">
        <div>
          <span class="list-title">${trade.label}</span>
          <a class="inline-link" href="${marketUrl(trade.market_url, trade.ticker)}" target="_blank" rel="noreferrer">Abrir mercado</a>
        </div>
        <span class="badge ${decisionClass(trade.side)}">${trade.side}</span>
      </div>
      <div class="meta">
        <span>${trade.contracts} contratos</span>
        <span>Entrada ${formatProbability(trade.entry_price)}</span>
        <span>Salida ${formatProbability(trade.exit_price)}</span>
        <span>${formatTime(trade.closed_at)}</span>
      </div>
      <div class="trade-summary">
        <span class="trade-pill">Cerrado</span>
        <span class="trade-pnl ${pnlClass(trade.pnl)}">${formatSignedMoney(trade.pnl)}</span>
      </div>
    </article>
  `).join("");
}

function setManualFeedback(text, kind = "info") {
  manualFeedback.textContent = text;
  manualFeedback.className = `manual-feedback ${kind}`;
}

async function sendManualOrder(ticker, side) {
  const contractsRaw = window.prompt(`¿Cuántos contratos quieres abrir en paper para ${ticker} ${side}?`, "5");
  if (contractsRaw === null) {
    return;
  }
  const contracts = Number.parseInt(contractsRaw, 10);
  if (!Number.isFinite(contracts) || contracts <= 0) {
    setManualFeedback("Número de contratos inválido.", "error");
    return;
  }
  const note = window.prompt("Nota opcional para esta acción manual:", "") || "";
  setManualFeedback("Enviando orden manual en paper mode...", "info");
  const payload = await fetchJson("/manual/paper-order", {
    method: "POST",
    body: JSON.stringify({ ticker, side, contracts, note }),
  });
  setManualFeedback(payload.message || "Orden creada.", "success");
  await refreshDashboard();
}

async function sendManualClose(tradeId) {
  const note = window.prompt("Nota opcional para este cierre manual:", "") || "";
  setManualFeedback("Cerrando trade manualmente en paper mode...", "info");
  const payload = await fetchJson("/manual/paper-close", {
    method: "POST",
    body: JSON.stringify({ trade_id: Number(tradeId), note }),
  });
  setManualFeedback(payload.message || "Trade cerrado.", "success");
  await refreshDashboard();
}

async function refreshDashboard() {
  refreshButton.disabled = true;
  try {
    const [state, live] = await Promise.all([
      fetchJson("/state?limit=12"),
      fetchJson("/live-markets?limit=80"),
    ]);

    const marketByTicker = new Map((live.markets || []).map((market) => [market.ticker, market]));

    renderDecisionSummary(state.signal_decisions || {});
    renderSkipSummary(state.skip_reason_summary || []);
    renderBlockedList(state.blocked_categories || []);
    renderMarkets(live.markets || [], state);
    renderSignals(state.recent_signals || []);
    const currentOpenPnl = renderOpenTrades(state.open_trades || [], marketByTicker, state.effective_mode);
    renderClosedTrades(state.recent_closed_trades || []);

    marketCount.textContent = String(live.count || 0);
    openCount.textContent = String(state.open_trades_count || 0);
    closedCount.textContent = String(state.closed_trades_count || 0);
    blockedCount.textContent = String((state.blocked_categories || []).length);
    effectiveMode.textContent = state.effective_mode || "--";
    renderMetricPnl(totalPnl, Number(state.total_pnl || 0) + currentOpenPnl);
    renderMetricPnl(realizedPnl, state.realized_pnl || 0);
    renderMetricPnl(openPnl, currentOpenPnl);
    lastRefresh.textContent = formatTime(live.timestamp || state.timestamp);
    const manualEnabled = Boolean(state.manual_actions_enabled ?? (state.effective_mode === "demo"));
    manualActionHint.textContent = manualEnabled
      ? "Paper mode activo. Los botones YES/NO abren posiciones manuales simuladas."
      : "Modo production detectado. Las acciones manuales quedan deshabilitadas desde este dashboard.";
    setStatus("ok", `Activo · ${live.latency_ms || 0} ms`);
  } catch (error) {
    setStatus("bad", "Sin lectura reciente");
    setManualFeedback(String(error.message || error), "error");
    console.error(error);
  } finally {
    refreshButton.disabled = false;
  }
}

document.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.dataset.manualSide && target.dataset.ticker) {
    try {
      await sendManualOrder(target.dataset.ticker, target.dataset.manualSide);
    } catch (error) {
      setManualFeedback(String(error.message || error), "error");
    }
  }
  if (target.dataset.closeTrade) {
    try {
      await sendManualClose(target.dataset.closeTrade);
    } catch (error) {
      setManualFeedback(String(error.message || error), "error");
    }
  }
});

refreshButton.addEventListener("click", refreshDashboard);
refreshDashboard();
window.setInterval(refreshDashboard, REFRESH_MS);
