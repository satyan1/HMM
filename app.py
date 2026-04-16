import hmac
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import fetch_hourly_data, compute_features
from hmm_engine import train_hmm, add_regimes_to_df
from indicators import add_indicators, compute_votes
from backtester import run_backtest, INITIAL_CAPITAL, LEVERAGE, REQUIRED_VOTES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HMM Regime Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Password gate ─────────────────────────────────────────────────────────────
def _check_password() -> bool:
    def _password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["_password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["_password_correct"] = False

    if st.session_state.get("_password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=_password_entered, key="password")
    if "_password_correct" in st.session_state:
        st.error("Incorrect password — try again.")
    return False

if not _check_password():
    st.stop()
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 6px 0;
    }
    .signal-long  { color: #00e676; font-size: 2rem; font-weight: 800; }
    .signal-cash  { color: #ff9800; font-size: 2rem; font-weight: 800; }
    .regime-bull  { color: #00e676; font-weight: 700; }
    .regime-bear  { color: #f44336; font-weight: 700; }
    .regime-other { color: #90caf9; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    days = st.slider("History (days)", 180, 730, 730, 30)
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Strategy Rules**")
    st.markdown(f"- Min votes required: **{REQUIRED_VOTES}/8**")
    st.markdown(f"- Leverage: **{LEVERAGE}×**")
    st.markdown("- Cooldown after exit: **48 h**")
    st.markdown("- Exit on: **Bear/Crash** regime")

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None

# ── Main pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data and training HMM…"):
        try:
            raw_df = fetch_hourly_data(ticker, days=days)
            feat_df = compute_features(raw_df)
            feat_df = add_indicators(feat_df)
            feat_df.dropna(inplace=True)

            feature_cols = ["Returns", "Range", "Vol_vol"]
            X = feat_df[feature_cols].values

            model = train_hmm(X)
            feat_df, regime_map, bull_state, bear_state = add_regimes_to_df(feat_df, model)

            results = run_backtest(feat_df, bull_state, bear_state)
            results["df"] = feat_df
            results["regime_map"] = regime_map
            results["bull_state"] = bull_state
            results["bear_state"] = bear_state
            results["ticker"] = ticker

            st.session_state["results"] = results
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ── Display ────────────────────────────────────────────────────────────────────
if st.session_state["results"] is None:
    st.info("Configure the ticker in the sidebar and click **Run Analysis** to begin.")
    st.stop()

res = st.session_state["results"]
df: pd.DataFrame = res["df"]
ticker_label = res["ticker"]

# ── Current signal / regime ────────────────────────────────────────────────────
last_row = df.iloc[-1]
current_regime = last_row["Regime"]
vote_score, vote_dict = compute_votes(last_row)

is_long = current_regime == "Bull Run" and vote_score >= REQUIRED_VOTES
signal_label = "LONG" if is_long else "CASH"
signal_class = "signal-long" if is_long else "signal-cash"

if current_regime == "Bull Run":
    regime_class = "regime-bull"
elif current_regime == "Bear/Crash":
    regime_class = "regime-bear"
else:
    regime_class = "regime-other"

st.markdown(f"## {ticker_label} — Regime Trading Dashboard")

col_sig, col_reg, col_votes = st.columns(3)
with col_sig:
    st.markdown("**Current Signal**")
    st.markdown(f'<span class="{signal_class}">{signal_label}</span>', unsafe_allow_html=True)
with col_reg:
    st.markdown("**Detected Regime**")
    st.markdown(f'<span class="{regime_class}">{current_regime}</span>', unsafe_allow_html=True)
with col_votes:
    st.markdown("**Vote Score**")
    st.markdown(
        f'<span style="font-size:2rem;font-weight:800;color:#90caf9">{vote_score}/8</span>',
        unsafe_allow_html=True,
    )

# Confirmation table
with st.expander("Confirmation Details", expanded=False):
    rows = [{"Condition": k, "Pass": "✅" if v else "❌"} for k, v in vote_dict.items()]
    st.table(pd.DataFrame(rows).set_index("Condition"))

st.divider()

# ── Metrics row ────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Return", f"{res['total_return']:.1f}%")
m2.metric("Alpha vs B&H", f"{res['alpha']:.1f}%")
m3.metric("Win Rate", f"{res['win_rate']:.1f}%")
m4.metric("Max Drawdown", f"{res['max_drawdown']:.1f}%")

extra1, extra2, extra3 = st.columns(3)
extra1.metric("# Trades", res["n_trades"])
extra2.metric("Final Equity", f"${res['final_equity']:,.0f}")
extra3.metric("B&H Return", f"{res['bnh_return']:.1f}%")

st.divider()

# ── Regime colour map ─────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Bull Run":    "rgba(0,230,118,0.12)",
    "Bear/Crash":  "rgba(244,67,54,0.14)",
    "Rally":       "rgba(0,200,83,0.07)",
    "Correction":  "rgba(255,82,82,0.07)",
    "Neutral":     "rgba(144,202,249,0.06)",
}
REGIME_LINE_COLORS = {
    "Bull Run":    "#00e676",
    "Bear/Crash":  "#f44336",
    "Rally":       "#69f0ae",
    "Correction":  "#ff5252",
    "Neutral":     "#90caf9",
}

# ── Candlestick chart ─────────────────────────────────────────────────────────
# Reduce to last 2000 bars for performance
plot_df = df.tail(2000).copy()

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.60, 0.20, 0.20],
    vertical_spacing=0.03,
    subplot_titles=("Price & Regimes", "Volume", "Equity Curve"),
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        showlegend=False,
    ),
    row=1, col=1,
)

# EMA lines
fig.add_trace(
    go.Scatter(x=plot_df.index, y=plot_df["EMA50"], name="EMA50",
               line=dict(color="#ffd54f", width=1), opacity=0.8),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=plot_df.index, y=plot_df["EMA200"], name="EMA200",
               line=dict(color="#ce93d8", width=1), opacity=0.8),
    row=1, col=1,
)

# Regime background shading (vrect per contiguous block)
regime_col = plot_df["Regime"]
prev_regime = None
block_start = None

def _add_vrect(fig, t0, t1, regime):
    color = REGIME_COLORS.get(regime, "rgba(128,128,128,0.05)")
    fig.add_vrect(
        x0=t0, x1=t1,
        fillcolor=color,
        opacity=1.0,
        layer="below",
        line_width=0,
        row=1, col=1,
    )

for ts, reg in regime_col.items():
    if reg != prev_regime:
        if prev_regime is not None:
            _add_vrect(fig, block_start, ts, prev_regime)
        block_start = ts
        prev_regime = reg
if prev_regime is not None:
    _add_vrect(fig, block_start, regime_col.index[-1], prev_regime)

# Trade markers
eq_df = res["equity_curve"]
trade_df = res["trades"]
if len(trade_df) > 0:
    # Filter to visible range
    visible_trades = trade_df[
        (trade_df["entry_time"] >= plot_df.index[0]) |
        (trade_df["exit_time"] >= plot_df.index[0])
    ]
    fig.add_trace(
        go.Scatter(
            x=visible_trades["entry_time"],
            y=visible_trades["entry_price"],
            mode="markers",
            name="Entry",
            marker=dict(symbol="triangle-up", color="#00e676", size=10),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=visible_trades["exit_time"],
            y=visible_trades["exit_price"],
            mode="markers",
            name="Exit",
            marker=dict(symbol="triangle-down", color="#f44336", size=10),
        ),
        row=1, col=1,
    )

# Volume bars
vol_colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(plot_df["Close"], plot_df["Open"])]
fig.add_trace(
    go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume",
           marker_color=vol_colors, showlegend=False),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["Volume"].rolling(20).mean(),
        name="Vol SMA20",
        line=dict(color="#ffd54f", width=1),
    ),
    row=2, col=1,
)

# Equity curve
eq_plot = eq_df[eq_df.index >= plot_df.index[0]]
fig.add_trace(
    go.Scatter(
        x=eq_plot.index,
        y=eq_plot["equity"],
        name="Equity",
        line=dict(color="#42a5f5", width=2),
        fill="tozeroy",
        fillcolor="rgba(66,165,245,0.08)",
    ),
    row=3, col=1,
)

fig.update_layout(
    template="plotly_dark",
    height=820,
    margin=dict(t=40, b=40, l=60, r=20),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Inter, sans-serif", size=12),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
)
fig.update_yaxes(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e")
fig.update_xaxes(gridcolor="#1e1e2e")

st.plotly_chart(fig, use_container_width=True)

# ── Regime legend ────────────────────────────────────────────────────────────
st.markdown("**Regime Legend**")
leg_cols = st.columns(len(REGIME_COLORS))
for col, (reg, color) in zip(leg_cols, REGIME_LINE_COLORS.items()):
    col.markdown(
        f'<span style="color:{color};font-weight:600">■ {reg}</span>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Trade log ──────────────────────────────────────────────────────────────────
st.subheader("Trade Log")
if len(trade_df) == 0:
    st.info("No trades generated in this period.")
else:
    display_trades = trade_df.copy()
    display_trades["pnl"] = display_trades["pnl"].map("${:,.2f}".format)
    display_trades["entry_price"] = display_trades["entry_price"].map("{:.4f}".format)
    display_trades["exit_price"] = display_trades["exit_price"].map("{:.4f}".format)
    st.dataframe(display_trades, use_container_width=True)

# ── Equity curve standalone ────────────────────────────────────────────────────
st.subheader("Full Equity Curve vs Buy & Hold")
bnh_equity = INITIAL_CAPITAL * (
    1 + (df["Close"] / df["Close"].iloc[0] - 1) * LEVERAGE
)
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=eq_df.index, y=eq_df["equity"],
        name="Strategy", line=dict(color="#42a5f5", width=2),
    )
)
fig2.add_trace(
    go.Scatter(
        x=df.index, y=bnh_equity,
        name=f"B&H {LEVERAGE}×", line=dict(color="#ffd54f", width=1.5, dash="dash"),
    )
)
fig2.update_layout(
    template="plotly_dark",
    height=320,
    margin=dict(t=20, b=30, l=60, r=20),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    yaxis_title="Equity ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    font=dict(family="Inter, sans-serif", size=12),
)
fig2.update_yaxes(gridcolor="#1e1e2e")
fig2.update_xaxes(gridcolor="#1e1e2e")
st.plotly_chart(fig2, use_container_width=True)
