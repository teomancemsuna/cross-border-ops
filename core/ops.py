"""
Analytics engine for cross-border order operations.

No Streamlit code lives here: every function takes and returns plain pandas
objects, so the logic is easy to test and the app stays a thin UI layer.

Main ideas
    * Order status is DERIVED from dates with simple, documented rules,
      the same way an operations person would read a shipment report.
    * Exceptions get a rule-based next action, so the table is a work list,
      not just a report.
    * Weekly order volume is forecast with Holt's linear trend method, then a
      holiday uplift is applied and compared with warehouse prep capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TODAY = pd.Timestamp("2026-11-02")      # snapshot date of the synthetic data
LIKELY_LOST_DAYS = 21                   # no delivery this long after ordering -> likely lost
ESCALATE_DAYS_LATE = 5                  # days past promise before escalating with the carrier
HOLIDAY_WEEKS = (pd.Timestamp("2026-11-23"), pd.Timestamp("2026-12-14"))  # Black Friday -> mid Dec

STATUS_ORDER = ["Delivered", "In transit", "Late", "Likely lost", "Lost (refunded)"]
DATE_COLS = ["order_date", "ship_date", "promised_date", "delivered_date"]


# --------------------------------------------------------------------------- #
# Loading and status rules
# --------------------------------------------------------------------------- #
def prepare(raw: pd.DataFrame, today: pd.Timestamp = TODAY) -> pd.DataFrame:
    """
    Add derived columns. Status rules (evaluated at `today`):
        Delivered    - a delivery date before today
        In transit   - not delivered, promise date not passed yet
        Late         - not delivered, promise date passed
        Likely lost  - not delivered, ordered LIKELY_LOST_DAYS or more ago
        Lost (refunded) - a lost parcel that has already been resolved
    """
    df = raw.copy()
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c])

    # A delivery dated today or later has not happened yet at snapshot time.
    df.loc[df["delivered_date"] >= today, "delivered_date"] = pd.NaT

    delivered = df["delivered_date"].notna()
    age = (today - df["order_date"]).dt.days
    past_promise = today > df["promised_date"]

    refunded = df["refunded"].astype(bool) if "refunded" in df else pd.Series(False, index=df.index)
    df["status"] = np.select(
        [delivered, refunded, ~past_promise, age >= LIKELY_LOST_DAYS],
        ["Delivered", "Lost (refunded)", "In transit", "Likely lost"],
        default="Late",
    )
    df["age_days"] = age
    df["delivery_days"] = (df["delivered_date"] - df["order_date"]).dt.days
    df["on_time"] = delivered & (df["delivered_date"] <= df["promised_date"])
    end = df["delivered_date"].fillna(today)
    df["days_late"] = (end - df["promised_date"]).dt.days.clip(lower=0)
    df["week"] = df["order_date"].dt.to_period("W-SUN").dt.start_time
    return df


def filter_orders(df: pd.DataFrame, start=None, end=None, routes=None, provinces=None) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["order_date"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["order_date"] <= pd.Timestamp(end)
    if routes:
        m &= df["route"].isin(routes)
    if provinces:
        m &= df["province"].isin(provinces)
    return df[m]


# --------------------------------------------------------------------------- #
# KPIs and breakdowns
# --------------------------------------------------------------------------- #
def kpis(df: pd.DataFrame) -> dict:
    d = df[df["status"] == "Delivered"]
    old_enough = df[df["age_days"] >= LIKELY_LOST_DAYS]
    return {
        "orders": len(df),
        "delivered": len(d),
        "on_time_rate": d["on_time"].mean() if len(d) else np.nan,
        "avg_delivery_days": d["delivery_days"].mean() if len(d) else np.nan,
        "p90_delivery_days": d["delivery_days"].quantile(0.9) if len(d) else np.nan,
        "open_exceptions": int(df["status"].isin(["Late", "Likely lost"]).sum()),
        "lost_rate": old_enough["status"].isin(["Likely lost", "Lost (refunded)"]).mean()
                     if len(old_enough) else np.nan,
        "avg_shipping_cost": df["shipping_cost_usd"].mean() if len(df) else np.nan,
    }


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per route: volume, speed, reliability and cost."""
    d = df[df["status"] == "Delivered"]
    g = d.groupby("route")
    out = pd.DataFrame({
        "on_time_rate": g["on_time"].mean(),
        "avg_days": g["delivery_days"].mean(),
        "p90_days": g["delivery_days"].quantile(0.9),
    })
    all_g = df.groupby("route")
    out["orders"] = all_g.size()
    out["share"] = out["orders"] / out["orders"].sum()
    out["avg_cost_usd"] = all_g["shipping_cost_usd"].mean()
    out["late_or_lost"] = all_g["status"].apply(lambda s: s.isin(["Late", "Likely lost"]).sum())
    return out.reset_index().sort_values("orders", ascending=False)


def province_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["status"] == "Delivered"]
    g = d.groupby("province")
    return pd.DataFrame({
        "orders": g.size(),
        "on_time_rate": g["on_time"].mean(),
        "avg_days": g["delivery_days"].mean(),
    }).reset_index().sort_values("on_time_rate")


def weekly_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Orders per order-week and the on-time rate of those orders once delivered."""
    g = df.groupby("week")
    out = pd.DataFrame({"orders": g.size()})
    d = df[df["status"] == "Delivered"].groupby("week")
    out["on_time_rate"] = d["on_time"].mean()
    out["delivered_share"] = d.size() / out["orders"]
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Exceptions work list
# --------------------------------------------------------------------------- #
def exceptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Open orders that need someone to act, with a rule-based next step:
        P1  Likely lost             -> file a carrier claim; refund or reship
        P2  Late >= 5 days          -> escalate with carrier; send customer a new ETA
        P3  Late 1-4 days           -> send a proactive delay message
    """
    ex = df[df["status"].isin(["Late", "Likely lost"])].copy()
    lost = ex["status"] == "Likely lost"
    big = ex["days_late"] >= ESCALATE_DAYS_LATE
    ex["priority"] = np.select([lost, big], ["P1", "P2"], default="P3")
    ex["action"] = np.select(
        [lost, big],
        ["File carrier claim; refund or reship",
         "Escalate with carrier; send customer new ETA"],
        default="Send proactive delay message",
    )
    cols = ["priority", "order_id", "order_date", "promised_date", "days_late", "age_days",
            "route", "province", "category", "sale_price_cad", "status", "action"]
    return ex[cols].sort_values(["priority", "days_late"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
def complete_weeks(df: pd.DataFrame) -> pd.Series:
    """Weekly order counts, dropping a partial first or last week."""
    counts = df.groupby("week").size()
    days_seen = df.groupby("week")["order_date"].nunique()
    return counts[days_seen == 7]


def holt_fit(y: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Holt's linear trend (double exponential smoothing):
        level_t = a * y_t + (1 - a) * (level_{t-1} + trend_{t-1})
        trend_t = b * (level_t - level_{t-1}) + (1 - b) * trend_{t-1}
    a and b are chosen by grid search to minimise one-step-ahead squared error.
    Returns (alpha, beta, last_level, last_trend, residual_std).
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        raise ValueError("Need at least 4 weeks of history to forecast.")
    best = None
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    for a in grid:
        for b in grid:
            level, trend = y[0], y[1] - y[0]
            errs = []
            for t in range(1, len(y)):
                pred = level + trend
                errs.append(y[t] - pred)
                new_level = a * y[t] + (1 - a) * pred
                trend = b * (new_level - level) + (1 - b) * trend
                level = new_level
            sse = float(np.sum(np.square(errs)))
            if best is None or sse < best[0]:
                best = (sse, a, b, level, trend, float(np.std(errs[1:], ddof=1)))
    _, a, b, level, trend, sd = best
    return a, b, level, trend, sd


def forecast_weekly(history: pd.Series, horizon: int = 8, uplift_pct: float = 0.0,
                    z: float = 1.28) -> pd.DataFrame:
    """
    Forecast `horizon` weeks after the last week in `history` (index = week start).
    The 80% interval (z = 1.28) widens with sqrt(weeks ahead).
    Holiday weeks get the uplift on top of the trend forecast.
    """
    a, b, level, trend, sd = holt_fit(history.values)
    start = history.index[-1] + pd.Timedelta(weeks=1)
    weeks = pd.date_range(start, periods=horizon, freq="7D")
    h = np.arange(1, horizon + 1)
    base = level + h * trend
    holiday = (weeks >= HOLIDAY_WEEKS[0]) & (weeks <= HOLIDAY_WEEKS[1])
    factor = np.where(holiday, 1 + uplift_pct / 100.0, 1.0)
    spread = z * sd * np.sqrt(h)
    return pd.DataFrame({
        "week": weeks,
        "baseline": base,
        "forecast": base * factor,
        "low": np.maximum(0, (base - spread) * factor),
        "high": (base + spread) * factor,
        "is_holiday": holiday,
    })


def capacity_gaps(fc: pd.DataFrame, capacity_per_week: float) -> pd.DataFrame:
    """Weeks where forecast (or the high end of the interval) exceeds prep capacity."""
    out = fc.copy()
    out["gap"] = out["forecast"] - capacity_per_week
    out["gap_high"] = out["high"] - capacity_per_week
    out["over"] = out["gap"] > 0
    out["at_risk"] = (~out["over"]) & (out["gap_high"] > 0)
    return out
