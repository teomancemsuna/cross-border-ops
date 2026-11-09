"""
Cross-Border Order Operations — Streamlit app.

A seller ships marketplace orders from a US warehouse to customers in Canada
through three carrier routes. This dashboard answers three daily questions:
    1. Overview   — Are we delivering on our promise, and which route/region is slipping?
    2. Exceptions — Which orders need someone to act today, and what should they do?
    3. Forecast   — Will next weeks' volume (incl. Black Friday) fit our warehouse capacity?

Remember: Streamlit reruns this file top to bottom on every interaction.
All calculations live in core/ops.py; this file only filters, calls and draws.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.ops import (
    HOLIDAY_WEEKS, TODAY, capacity_gaps, complete_weeks, exceptions, filter_orders,
    forecast_weekly, kpis, prepare, route_summary, weekly_performance,
)

DATA_PATH = Path(__file__).parent / "data" / "orders.csv"
ON_TIME_TARGET = 0.95

TEAL, INK, MUTED, GRID = "#0F766E", "#111827", "#6B7280", "#E5E7EB"
ROUTE_COLORS = {"Asendia → Canada Post": "#2563EB", "Purolator (end-to-end)": "#0F766E",
                "Asendia → Intelcom": "#D97706"}
PRIORITY_COLORS = {"P1": "#DC2626", "P2": "#D97706", "P3": "#2563EB"}

st.set_page_config(page_title="Cross-Border Order Ops", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.kpi {background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:14px 16px;
      box-shadow:0 1px 2px rgba(16,24,40,.04);}
.kpi .label {color:#6B7280; font-size:.78rem; font-weight:600;}
.kpi .value {color:#111827; font-size:1.6rem; font-weight:700; margin-top:2px;}
.kpi .sub {font-size:.8rem; margin-top:2px; color:#6B7280;}
.note {background:#FFFFFF; border:1px solid #E5E7EB; border-radius:10px; padding:10px 14px;
       margin-bottom:10px; font-size:.9rem; color:#1F2937;}
.badge {display:inline-block; white-space:nowrap; padding:3px 10px; border-radius:999px; font-size:.75rem;
        font-weight:600; background:#FEF3C7; color:#92400E; border:1px solid #FCD34D;}
h4 {margin-bottom: .2rem;}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", color=MUTED):
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub" style="color:{color}">{sub}</div></div>')


def note(html, color=TEAL):
    st.markdown(f'<div class="note" style="border-left:4px solid {color}">{html}</div>',
                unsafe_allow_html=True)


def style(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.12, x=0), font=dict(color=INK))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def load() -> pd.DataFrame:
    return prepare(pd.read_csv(DATA_PATH))


orders = load()

# --------------------------------------------------------------------------- #
# Sidebar filters (apply to Overview and Exceptions)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Filters")
    dmin, dmax = orders.order_date.min().date(), orders.order_date.max().date()
    date_range = st.date_input("Order date", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    routes = st.multiselect("Carrier route", sorted(orders.route.unique()))
    provinces = st.multiselect("Province", sorted(orders.province.unique()))
    st.caption("Empty selection = all. The forecast always uses every order.")
    st.divider()
    st.caption(f"Snapshot: {TODAY:%d %b %Y}. Synthetic data — demonstration only.")

# date_input returns a single date while the user is still picking the second one
start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (dmin, dmax))
df = filter_orders(orders, start, end, routes, provinces)
if df.empty:
    st.warning("No orders match these filters.")
    st.stop()

k = kpis(df)
ex = exceptions(df)
rs = route_summary(df)

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 1])
left.markdown("## Cross-Border Order Operations")
left.caption(f"US warehouse → customers in Canada · {k['orders']:,} orders "
             f"({start:%d %b} – {end:%d %b %Y}) · promise window 12 days")
right.markdown('<div style="text-align:right;margin-top:18px"><span class="badge">'
               'SYNTHETIC DATA — DEMONSTRATION ONLY</span></div>', unsafe_allow_html=True)

c = st.columns(6)
ot_color = TEAL if k["on_time_rate"] >= ON_TIME_TARGET else PRIORITY_COLORS["P1"]
p1 = int((ex.priority == "P1").sum())
c[0].markdown(kpi("Orders", f"{k['orders']:,}", f"{k['delivered']:,} delivered"), unsafe_allow_html=True)
c[1].markdown(kpi("On-time delivery", f"{k['on_time_rate']:.1%}",
                  f"target {ON_TIME_TARGET:.0%}", ot_color), unsafe_allow_html=True)
c[2].markdown(kpi("Delivery time", f"{k['avg_delivery_days']:.1f} days",
                  f"90% within {k['p90_delivery_days']:.0f} days"), unsafe_allow_html=True)
c[3].markdown(kpi("Open exceptions", f"{k['open_exceptions']}",
                  f"{p1} need a claim (P1)", PRIORITY_COLORS["P1"] if p1 else TEAL), unsafe_allow_html=True)
c[4].markdown(kpi("Lost parcel rate", f"{k['lost_rate']:.1%}", "orders 21+ days old"), unsafe_allow_html=True)
c[5].markdown(kpi("Avg shipping cost", f"${k['avg_shipping_cost']:.2f}", "USD per order"),
              unsafe_allow_html=True)
st.write("")

tab_overview, tab_ex, tab_fc = st.tabs(["Overview", f"Exceptions ({len(ex)})", "Forecast & capacity"])

# --------------------------------------------------------------------------- #
# Tab 1 — Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("#### Weekly volume and on-time rate")
        w = weekly_performance(df)
        full = complete_weeks(df).index
        w = w[w.week.isin(full)]                       # drop partial first/last weeks
        # Recent weeks still have parcels in transit, so their on-time rate is not final yet.
        w_final = w[w.delivered_share >= 0.9]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=w.week, y=w.orders, name="Orders", marker_color="#CBD5E1"))
        fig.add_trace(go.Scatter(x=w_final.week, y=w_final.on_time_rate * 100, name="On-time %",
                                 line=dict(color=TEAL, width=3), mode="lines+markers"), secondary_y=True)
        fig.add_hline(y=ON_TIME_TARGET * 100, line=dict(color=PRIORITY_COLORS["P1"], dash="dot"),
                      secondary_y=True, annotation_text="95% target", annotation_position="bottom right")
        fig.update_yaxes(title_text="Orders / week", secondary_y=False)
        fig.update_yaxes(title_text="On-time %", range=[70, 101], tickmode="linear", tick0=70, dtick=5,
                         secondary_y=True, showgrid=False)
        st.plotly_chart(style(fig), width="stretch")
        st.caption("On-time % is shown only for weeks where at least 90% of orders have been delivered.")

    with col_b:
        st.markdown("#### Route trade-off: cost vs reliability")
        fig2 = go.Figure()
        for r in rs.itertuples():
            fig2.add_trace(go.Scatter(
                x=[r.avg_cost_usd], y=[r.on_time_rate * 100], mode="markers+text", name=r.route,
                text=[f"{r.route}<br>{r.share:.0%} of orders"], textposition="top center",
                marker=dict(size=18 + 40 * r.share, color=ROUTE_COLORS.get(r.route, TEAL), opacity=.85),
                hovertemplate=(f"{r.route}<br>${r.avg_cost_usd:.2f}/order<br>"
                               f"{r.on_time_rate:.1%} on time<br>{r.avg_days:.1f} days avg<extra></extra>")))
        fig2.update_xaxes(title_text="Avg shipping cost (USD / order)", range=[6.5, 16.5])
        fig2.update_yaxes(title_text="On-time %", range=[84, 106])
        fig2.update_layout(showlegend=False)
        st.plotly_chart(style(fig2), width="stretch")

    col_c, col_d = st.columns([3, 2], gap="large")
    with col_c:
        st.markdown("#### On-time rate by route and province")
        d = df[df.status == "Delivered"]
        pivot = d.pivot_table(index="route", columns="province", values="on_time", aggfunc="mean")
        counts = d.pivot_table(index="route", columns="province", values="on_time", aggfunc="size")
        pivot = pivot.where(counts >= 20)               # hide cells with too few orders
        order_cols = d.groupby("province").size().sort_values(ascending=False).index
        pivot = pivot[[p for p in order_cols if p in pivot.columns]]
        heat = go.Figure(go.Heatmap(
            z=pivot.values * 100, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#DC2626"], [0.5, "#FBBF24"], [1, "#0F766E"]], zmin=80, zmax=100,
            text=[["" if np.isnan(v) else f"{v * 100:.0f}%" for v in row] for row in pivot.values],
            texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{z:.1f}% on time<extra></extra>",
            colorbar=dict(title="On-time %")))
        st.plotly_chart(style(heat, 260), width="stretch")
        st.caption("Cells with fewer than 20 delivered orders are left blank.")

    with col_d:
        st.markdown("#### What this means")
        best = rs.sort_values("on_time_rate", ascending=False).iloc[0]
        cheapest = rs.sort_values("avg_cost_usd").iloc[0]
        if cheapest.route != best.route:
            note(f"<b>{cheapest.route}</b> is the cheapest route (${cheapest.avg_cost_usd:.2f}/order) but "
                 f"delivers on time only <b>{cheapest.on_time_rate:.0%}</b>. "
                 f"<b>{best.route}</b> costs ${best.avg_cost_usd - cheapest.avg_cost_usd:.2f} more "
                 f"and is on time {best.on_time_rate:.1%}.", PRIORITY_COLORS["P2"])
        stacked = pd.DataFrame({"rate": pivot.stack(), "n": counts.stack()}).dropna()
        if len(stacked):
            worst_idx = stacked["rate"].idxmin()
            worst = stacked.loc[worst_idx]
            if worst.rate < ON_TIME_TARGET:
                note(f"Weakest lane: <b>{worst_idx[0]}</b> to <b>{worst_idx[1]}</b> at "
                     f"<b>{worst.rate:.0%}</b> on time ({int(worst.n)} orders). Consider routing "
                     f"{worst_idx[1]} orders through {best.route}.", PRIORITY_COLORS["P1"])
        if len(ex):
            value = ex.loc[ex.priority == "P1", "sale_price_cad"].sum()
            note(f"<b>{p1} parcels</b> are likely lost, worth <b>C${value:,.0f}</b> in sales. "
                 f"See the Exceptions tab for the claim list.", PRIORITY_COLORS["P1"])

    st.markdown("#### Route summary")
    rs_view = rs.assign(share=rs.share * 100, on_time_rate=rs.on_time_rate * 100)
    st.dataframe(
        rs_view.rename(columns={"route": "Route", "orders": "Orders", "share": "Share",
                           "on_time_rate": "On time", "avg_days": "Avg days", "p90_days": "90% within (days)",
                           "avg_cost_usd": "Avg cost (USD)", "late_or_lost": "Open late / lost"}),
        hide_index=True, width="stretch",
        column_config={"Share": st.column_config.NumberColumn(format="%.0f%%"),
                       "On time": st.column_config.NumberColumn(format="%.1f%%"),
                       "Avg days": st.column_config.NumberColumn(format="%.1f"),
                       "90% within (days)": st.column_config.NumberColumn(format="%.0f"),
                       "Avg cost (USD)": st.column_config.NumberColumn(format="$%.2f")})

# --------------------------------------------------------------------------- #
# Tab 2 — Exceptions work list
# --------------------------------------------------------------------------- #
with tab_ex:
    st.markdown("#### Orders that need action today")
    st.caption("P1 likely lost (21+ days, not delivered) · P2 five or more days past promise · "
               "P3 one to four days past promise")
    pc = st.columns(3)
    for i, p in enumerate(["P1", "P2", "P3"]):
        sub = ex[ex.priority == p]
        pc[i].markdown(kpi(f"{p} · {sub.action.iloc[0] if len(sub) else '—'}", f"{len(sub)}",
                           f"C${sub.sale_price_cad.sum():,.0f} in sales", PRIORITY_COLORS[p]),
                       unsafe_allow_html=True)
    st.write("")
    pick = st.multiselect("Show priority", ["P1", "P2", "P3"], default=["P1", "P2", "P3"])
    view = ex[ex.priority.isin(pick)]
    st.dataframe(
        view.drop(columns=["status", "age_days"]).style.map(lambda v: f"color:{PRIORITY_COLORS.get(v, INK)}; font-weight:700", subset=["priority"]),
        hide_index=True, width="stretch",
        column_config={"order_date": st.column_config.DateColumn("Ordered", format="DD MMM"),
                       "promised_date": st.column_config.DateColumn("Promised", format="DD MMM"),
                       "days_late": "Days late", "age_days": "Age (days)",
                       "sale_price_cad": st.column_config.NumberColumn("Sale (CAD)", format="$%.2f"),
                       "priority": "Priority", "order_id": "Order", "route": "Route",
                       "province": "Prov.", "category": "Category", "status": "Status",
                       "action": "Next action"})
    st.download_button("Download work list (CSV)", view.to_csv(index=False).encode(),
                       file_name=f"exceptions_{TODAY:%Y%m%d}.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Tab 3 — Forecast & warehouse capacity
# --------------------------------------------------------------------------- #
with tab_fc:
    st.markdown("#### Next 8 weeks: order volume vs warehouse prep capacity")
    s1, s2 = st.columns(2)
    uplift = s1.slider("Holiday uplift, Black Friday to mid-December (%)", 0, 80, 30, step=5)
    capacity = s2.slider("Warehouse prep capacity (orders / week)", 200, 450, 300, step=10)

    hist = complete_weeks(orders)
    fc = capacity_gaps(forecast_weekly(hist, horizon=8, uplift_pct=uplift), capacity)
    recent = hist.iloc[-12:]

    fig3 = go.Figure()
    fig3.add_vrect(x0=HOLIDAY_WEEKS[0] - pd.Timedelta(days=3), x1=HOLIDAY_WEEKS[1] + pd.Timedelta(days=3),
                   fillcolor="#FEF3C7", opacity=.6, line_width=0, layer="below",
                   annotation_text="Holiday peak", annotation_position="top left")
    fig3.add_trace(go.Bar(x=recent.index, y=recent.values, name="Actual orders", marker_color="#CBD5E1"))
    fig3.add_trace(go.Scatter(x=fc.week, y=fc.high, mode="lines", line=dict(width=0), showlegend=False,
                              hoverinfo="skip"))
    fig3.add_trace(go.Scatter(x=fc.week, y=fc.low, mode="lines", line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(15,118,110,0.15)", name="80% range", hoverinfo="skip"))
    fig3.add_trace(go.Scatter(x=fc.week, y=fc.forecast, name="Forecast", mode="lines+markers",
                              line=dict(color=TEAL, width=3)))
    fig3.add_hline(y=capacity, line=dict(color=PRIORITY_COLORS["P1"], dash="dash"),
                   annotation_text=f"Capacity {capacity}/week", annotation_position="top left")
    fig3.update_yaxes(title_text="Orders / week", rangemode="tozero")
    fig3.update_xaxes(tickformat="%d %b")
    fig3.update_layout(legend_traceorder="normal")
    st.plotly_chart(style(fig3, 380), width="stretch")

    over, risky = fc[fc.over], fc[fc.at_risk]
    n1, n2 = st.columns(2)
    with n1:
        if len(over):
            first, worst = over.iloc[0], over.loc[over.gap.idxmax()]
            note(f"<b>{len(over)} week(s) over capacity</b>, starting the week of <b>{first.week:%d %b}</b>. "
                 f"Peak shortfall is <b>{worst.gap:.0f} orders</b> in the week of {worst.week:%d %b} "
                 f"(about {worst.gap / 6:.0f} extra orders per working day).", PRIORITY_COLORS["P1"])
        else:
            note("The forecast stays within warehouse capacity for all 8 weeks.", TEAL)
        if len(risky):
            note(f"{len(risky)} more week(s) fit on the central forecast but exceed capacity at the high "
                 f"end of the range ({', '.join(f'{d:%d %b}' for d in risky.week)}).", PRIORITY_COLORS["P2"])
    with n2:
        table = fc[["week", "forecast", "low", "high", "gap"]].copy()
        table["status"] = np.select([fc.over, fc.at_risk], ["Over", "At risk"], default="OK")
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"week": st.column_config.DateColumn("Week of", format="DD MMM"),
                                    "forecast": st.column_config.NumberColumn("Forecast", format="%.0f"),
                                    "low": st.column_config.NumberColumn("Low", format="%.0f"),
                                    "high": st.column_config.NumberColumn("High", format="%.0f"),
                                    "gap": st.column_config.NumberColumn("vs capacity", format="%+.0f"),
                                    "status": "Status"})

    with st.expander("How the forecast works"):
        st.markdown("""
- **Model:** Holt's linear trend method (double exponential smoothing) on complete weeks of orders.
  The smoothing weights are picked by grid search to minimise one-week-ahead error.
- **Range:** 80% interval from the spread of past one-week-ahead errors, widening with the square root of weeks ahead.
- **Holiday uplift:** the trend model has not seen a holiday season yet (data starts in May), so the peak is an
  explicit, adjustable assumption applied to the weeks of 23 Nov – 14 Dec.
- **Capacity:** a week is *Over* when the forecast exceeds capacity and *At risk* when only the high end does.
""")
