import streamlit as st, pandas as pd, numpy as np
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(page_title="Market Risk & Capital Engine", layout="wide")
ASSETS = ["SPY", "TLT", "GLD", "USO", "EURUSD=X"]

@st.cache_data
def load():
    px = pd.read_csv("prices.csv", index_col=0, parse_dates=True).dropna()
    px = px[ASSETS]
    return np.log(px / px.shift(1)).dropna()

rets = load()

# ---------- sidebar ----------
st.sidebar.header("Portfolio")
notional = st.sidebar.number_input("Notional ($)", value=100_000_000, step=10_000_000)
default = {"SPY": 40, "TLT": 25, "GLD": 15, "USO": 10, "EURUSD=X": 10}
raw = pd.Series({a: float(st.sidebar.slider(f"{a} weight (%)", 0, 100, default[a])) for a in ASSETS})
if raw.sum() == 0:
    st.error("Set at least one weight above 0.")
    st.stop()
weights = raw / raw.sum()
st.sidebar.caption(f"Weights normalised to 100% (input sum {raw.sum():.0f}%)")
window = st.sidebar.slider("Estimation window (days)", 250, 1000, 500, 50)
limit = st.sidebar.number_input("1-day 99% VaR limit ($)", value=1_900_000, step=100_000)

port = rets @ weights
loss = -port
sq10 = np.sqrt(10)

# ---------- helpers ----------
def zone(x):
    return "Green" if x <= 4 else ("Yellow" if x <= 9 else "Red")

def plus_factor(x):
    if pd.isna(x):
        return np.nan
    t = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
    return 0.0 if x <= 4 else (1.0 if x >= 10 else t[int(x)])

def kupiec(x, n, p=0.01):
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0)
    lr = -2 * (ll0 - ll1)
    return 1 - stats.chi2.cdf(lr, 1)

@st.cache_data
def backtest(loss, window):
    port_ = -loss
    vh = loss.rolling(window).apply(lambda x: np.percentile(x, 99), raw=True).shift(1)
    vp = (-(port_.rolling(window).mean() + port_.rolling(window).std() * stats.norm.ppf(0.01))).shift(1)
    bt = pd.DataFrame({"loss": loss, "var_hist": vh, "var_param": vp}).dropna()
    for m in ["hist", "param"]:
        bt[f"exc_{m}"] = (bt["loss"] > bt[f"var_{m}"]).astype(int)
        bt[f"roll_{m}"] = bt[f"exc_{m}"].rolling(250).sum()
    return bt

def capital(bt, m):
    v10 = bt[f"var_{m}"] * sq10 * notional
    avg60 = v10.rolling(60).mean()
    mult = 3 + bt[f"roll_{m}"].apply(plus_factor)
    cap = pd.concat([v10.shift(1), mult * avg60], axis=1).max(axis=1)
    return pd.DataFrame({"mult": mult, "capital": cap}).dropna()

def metrics(w, sample):
    l = -(sample @ w)
    var99 = np.percentile(l, 99)
    v975 = np.percentile(l, 97.5)
    return var99 * notional, l[l >= v975].mean() * notional

bt = backtest(loss, window)
cap_h, cap_p = capital(bt, "hist"), capital(bt, "param")

# ---------- layout ----------
st.title("Market Risk & Regulatory Capital Engine")
st.caption("Educational implementation on liquid ETF/FX proxies. Not a compliant FRTB engine.")
t1, t2, t3, t4 = st.tabs(["1. Risk metrics", "2. Back-testing", "3. Capital charge", "4. New-position what-if"])

with t1:
    var_h = np.percentile(loss, 99)
    var_p = -(port.mean() + port.std() * stats.norm.ppf(0.01))
    sim = np.random.default_rng(42).multivariate_normal(rets.mean().values, rets.cov().values, 100_000) @ weights.values
    var_m = -np.percentile(sim, 1)
    v975 = np.percentile(loss, 97.5)
    es = loss[loss >= v975].mean()
    res = pd.DataFrame({"1-day loss (%)": [var_h, var_p, var_m, es]},
                       index=["Historical VaR 99%", "Parametric VaR 99%", "Monte Carlo VaR 99%", "Expected Shortfall 97.5%"])
    res["1-day ($)"] = res["1-day loss (%)"] * notional
    res["10-day ($)"] = res["1-day ($)"] * sq10
    res["1-day loss (%)"] *= 100
    st.dataframe(res.round(2), use_container_width=True)
    st.caption("Historical VaR exceeds the normal-based methods: real returns have fat tails.")
    cov = rets.cov().values
    w = weights.values
    contrib = w * (cov @ w) / (w @ cov @ w)
    fig = go.Figure()
    fig.add_bar(x=ASSETS, y=weights.values * 100, name="Weight %")
    fig.add_bar(x=ASSETS, y=contrib * 100, name="Risk contribution %")
    fig.update_layout(barmode="group", title="Weight vs risk contribution", height=380)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    n = len(bt)
    cols = st.columns(2)
    for col, (m, label) in zip(cols, [("hist", "Historical VaR"), ("param", "Parametric VaR")]):
        x = int(bt[f"exc_{m}"].sum())
        col.subheader(label)
        col.metric("Exceptions", f"{x}", f"expected ~{0.01 * n:.0f}", delta_color="off")
        col.metric("Kupiec p-value", f"{kupiec(x, n):.3f}", "Rejected" if kupiec(x, n) < 0.05 else "Not rejected", delta_color="off")
        col.metric("Latest 250-day zone", f"{zone(bt[f'roll_{m}'].iloc[-1])} ({int(bt[f'roll_{m}'].iloc[-1])})")
        col.metric("Worst 250-day window", f"{zone(bt[f'roll_{m}'].max())} ({int(bt[f'roll_{m}'].max())})")
    f1 = go.Figure()
    f1.add_scatter(x=bt.index, y=bt["loss"] * 100, name="Daily loss", line=dict(color="grey", width=1))
    f1.add_scatter(x=bt.index, y=bt["var_hist"] * 100, name="Historical VaR", line=dict(color="red"))
    f1.add_scatter(x=bt.index, y=bt["var_param"] * 100, name="Parametric VaR", line=dict(color="blue"))
    e = bt[bt["exc_hist"] == 1]
    f1.add_scatter(x=e.index, y=e["loss"] * 100, mode="markers", name="Hist. exceptions", marker=dict(color="black", size=6))
    f1.update_layout(title="Loss vs 99% VaR (out-of-sample)", yaxis_title="% loss", height=400)
    st.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure()
    f2.add_scatter(x=bt.index, y=bt["roll_hist"], name="Historical", line=dict(color="red"))
    f2.add_scatter(x=bt.index, y=bt["roll_param"], name="Parametric", line=dict(color="blue"))
    f2.add_hrect(y0=0, y1=4.5, fillcolor="green", opacity=0.12, line_width=0)
    f2.add_hrect(y0=4.5, y1=9.5, fillcolor="gold", opacity=0.18, line_width=0)
    f2.add_hrect(y0=9.5, y1=max(20, bt[["roll_hist", "roll_param"]].max().max() + 2), fillcolor="red", opacity=0.12, line_width=0)
    f2.update_layout(title="Basel traffic-light zones (exceptions in last 250 days)", height=380)
    st.plotly_chart(f2, use_container_width=True)

with t3:
    c = st.columns(2)
    c[0].metric("Latest capital: Historical VaR model", f"${cap_h['capital'].iloc[-1] / 1e6:.1f}M",
                f"multiplier {cap_h['mult'].iloc[-1]:.2f}", delta_color="off")
    c[1].metric("Latest capital: Parametric VaR model", f"${cap_p['capital'].iloc[-1] / 1e6:.1f}M",
                f"multiplier {cap_p['mult'].iloc[-1]:.2f}", delta_color="off")
    f3 = go.Figure()
    f3.add_scatter(x=cap_h.index, y=cap_h["capital"] / 1e6, name="Historical", line=dict(color="red"))
    f3.add_scatter(x=cap_p.index, y=cap_p["capital"] / 1e6, name="Parametric", line=dict(color="blue"))
    f3.update_layout(title="Illustrative capital = max(prior-day 10d VaR, (3 + plus factor) x 60-day avg 10d VaR)",
                     yaxis_title="$M", height=400)
    st.plotly_chart(f3, use_container_width=True)
    st.caption("Capital stays elevated for ~2 years after the March 2020 exceptions (procyclicality).")

with t4:
    oil = st.slider("Proposed USO weight (%)", 0, 60, 20) / 100
    basis = st.radio("Estimation sample", [f"Last {window} days", "Full history"], horizontal=True)
    sample = rets.iloc[-window:] if basis.startswith("Last") else rets
    w_new = weights.copy()
    others = [a for a in ASSETS if a != "USO"]
    if weights[others].sum() > 0:
        w_new[others] = weights[others] * (1 - oil) / weights[others].sum()
    w_new["USO"] = oil
    mult_now = cap_h["mult"].iloc[-1]
    rows = []
    for label, w_ in [("Current", weights), ("Proposed", w_new)]:
        v, e_ = metrics(w_, sample)
        rows.append([label, v, e_, v * sq10 * mult_now])
    out = pd.DataFrame(rows, columns=["Portfolio", "VaR 99% 1d ($)", "ES 97.5% 1d ($)", "Capital proxy ($)"]).set_index("Portfolio")
    chg = (out.loc["Proposed"] / out.loc["Current"] - 1) * 100
    out.loc["Change (%)"] = chg
    st.dataframe(out.round(1), use_container_width=True)
    st.write(f"Portfolio stays at ${notional / 1e6:.0f}M: other positions are scaled down pro-rata to fund the oil change.")
    breach = out.loc["Proposed", "VaR 99% 1d ($)"] > limit
    st.write(f"VaR limit ${limit:,.0f} breached by proposal: **{'YES' if breach else 'NO'}**")
    if chg["VaR 99% 1d ($)"] < 0 < chg["ES 97.5% 1d ($)"]:
        st.warning("VaR falls but Expected Shortfall rises: VaR ignores the size of losses beyond the "
                   "quantile. A VaR-only view would approve a trade that increases tail risk.")
