const statusLine = document.querySelector(".status-line");
const connectionStatus = document.getElementById("connection-status");
const refreshButton = document.getElementById("refresh-now");
const marketCount = document.getElementById("market-count");
const openCount = document.getElementById("open-count");
const blockedCount = document.getElementById("blocked-count");
const lastRefresh = document.getElementById("last-refresh");
const marketsBody = document.getElementById("markets-body");
const signalsList = document.getElementById("signals-list");
const tradesList = document.getElementById("trades-list");

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

function setStatus(kind, text) {
  statusLine.classList.remove("ok", "bad");
  if (kind) {
    statusLine.classList.add(kind);
  }
  connectionStatus.textContent = text;
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} ${response.status}`);
  }
  return response.json();
}

function renderMarkets(markets) {
  if (!markets.length) {
    marketsBody.innerHTML = '<tr><td colspan="7" class="empty">Sin mercados activos ahora</td></tr>';
    return;
  }

  marketsBody.innerHTML = markets.map((market) => {
    const assetClass = String(market.category || "").toLowerCase();
    return `
      <tr>
        <td class="ticker">${market.ticker}</td>
        <td><span class="pill ${assetClass}">${market.category}</span></td>
        <td>${market.strike ? formatMoney(market.strike) : "n/a"}</td>
        <td>${formatProbability(market.yes_ask)}</td>
        <td>${formatProbability(market.no_ask)}</td>
        <td>${formatNumber(market.volume_24h)}</td>
        <td>${formatDuration(market.time_to_expiry_s)}</td>
      </tr>
    `;
  }).join("");
}

function renderSignals(signals) {
  if (!signals.length) {
    signalsList.innerHTML = '<p class="empty">Sin senales todavia</p>';
    return;
  }

  signalsList.innerHTML = signals.slice().reverse().map((signal) => `
    <article class="list-item">
      <div class="list-top">
        <span class="list-title">${signal.ticker}</span>
        <span class="badge ${String(signal.decision).toLowerCase()}">${signal.decision}</span>
      </div>
      <div class="meta">
        <span>Delta ${formatProbability(signal.delta)}</span>
        <span>EV ${formatProbability(signal.ev_net_fees)}</span>
        <span>${signal.confidence}</span>
        <span>${formatTime(signal.timestamp)}</span>
      </div>
    </article>
  `).join("");
}

function renderTrades(trades) {
  if (!trades.length) {
    tradesList.innerHTML = '<p class="empty">Sin posiciones abiertas</p>';
    return;
  }

  tradesList.innerHTML = trades.map((trade) => `
    <article class="list-item">
      <div class="list-top">
        <span class="list-title">${trade.ticker}</span>
        <span class="badge ${String(trade.side).toLowerCase()}">${trade.side}</span>
      </div>
      <div class="meta">
        <span>${trade.contracts} contratos</span>
        <span>Entrada ${formatProbability(trade.entry_price)}</span>
        <span>${trade.mode}</span>
      </div>
    </article>
  `).join("");
}

async function refreshDashboard() {
  refreshButton.disabled = true;
  try {
    const [state, live] = await Promise.all([
      fetchJson("/state?limit=12"),
      fetchJson("/live-markets?limit=80"),
    ]);

    renderMarkets(live.markets || []);
    renderSignals(state.recent_signals || []);
    renderTrades(state.open_trades || []);

    marketCount.textContent = String(live.count || 0);
    openCount.textContent = String(state.open_trades_count || 0);
    blockedCount.textContent = String((state.blocked_categories || []).length);
    lastRefresh.textContent = formatTime(live.timestamp || state.timestamp);
    setStatus("ok", `Activo · ${live.latency_ms || 0} ms`);
  } catch (error) {
    setStatus("bad", "Sin lectura reciente");
    console.error(error);
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", refreshDashboard);
refreshDashboard();
window.setInterval(refreshDashboard, REFRESH_MS);
