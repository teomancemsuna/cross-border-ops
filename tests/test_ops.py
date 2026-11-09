"""Tests for the analytics engine. Run with:  pytest"""

import numpy as np
import pandas as pd
import pytest

from core.ops import (
    capacity_gaps, complete_weeks, exceptions, forecast_weekly, holt_fit, kpis, prepare,
)

TODAY = pd.Timestamp("2026-11-02")


def order(order_date, delivered=None, refunded=False, promise_days=12, **kw):
    od = pd.Timestamp(order_date)
    row = dict(order_id=kw.get("order_id", "T"), order_date=od, category="Toys", province="ON",
               route=kw.get("route", "R1"), ship_date=od + pd.Timedelta(days=1),
               promised_date=od + pd.Timedelta(days=promise_days),
               delivered_date=pd.Timestamp(delivered) if delivered else pd.NaT,
               sale_price_cad=30.0, shipping_cost_usd=10.0, refunded=refunded)
    return row


def frame(*rows):
    return prepare(pd.DataFrame(list(rows)), today=TODAY)


def test_status_rules():
    df = frame(
        order("2026-10-01", delivered="2026-10-09"),     # delivered
        order("2026-10-28"),                              # promise 9 Nov -> in transit
        order("2026-10-15"),                              # promise 27 Oct passed, 18 days old -> late
        order("2026-10-01"),                              # 32 days old -> likely lost
        order("2026-08-01", refunded=True),               # resolved loss
        order("2026-10-25", delivered="2026-11-05"),      # delivery after snapshot -> not yet delivered
    )
    assert df["status"].tolist() == ["Delivered", "In transit", "Late", "Likely lost",
                                     "Lost (refunded)", "In transit"]


def test_on_time_and_days_late():
    df = frame(
        order("2026-10-01", delivered="2026-10-13"),     # exactly on promise date -> on time
        order("2026-10-01", delivered="2026-10-16"),     # 3 days late
    )
    assert df["on_time"].tolist() == [True, False]
    assert df["days_late"].tolist() == [0, 3]
    assert kpis(df)["on_time_rate"] == pytest.approx(0.5)


def test_exception_priorities_and_actions():
    df = frame(
        order("2026-10-01", order_id="LOST"),            # likely lost
        order("2026-10-14", order_id="BIG"),             # promise 26 Oct, 7 days late
        order("2026-10-19", order_id="SMALL"),           # promise 31 Oct, 2 days late
        order("2026-10-30", order_id="FINE"),            # in transit, not an exception
    )
    ex = exceptions(df).set_index("order_id")
    assert set(ex.index) == {"LOST", "BIG", "SMALL"}
    assert ex.loc["LOST", "priority"] == "P1" and "claim" in ex.loc["LOST", "action"]
    assert ex.loc["BIG", "priority"] == "P2"
    assert ex.loc["SMALL", "priority"] == "P3"


def test_complete_weeks_drops_partial_weeks():
    days = pd.date_range("2026-10-01", "2026-10-25")     # Thu 1 Oct .. Sun 25 Oct
    df = frame(*[order(d, delivered=d + pd.Timedelta(days=5)) for d in days])
    w = complete_weeks(df)
    assert list(w.index) == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-12"),
                             pd.Timestamp("2026-10-19")]
    assert (w == 7).all()


def test_holt_follows_a_straight_line():
    y = 100 + 5 * np.arange(20)
    a, b, level, trend, sd = holt_fit(y)
    assert level == pytest.approx(195, abs=0.5)
    assert trend == pytest.approx(5, abs=0.2)


def test_forecast_uplift_only_in_holiday_weeks_and_interval_widens():
    hist = pd.Series(100 + 2 * np.arange(10) + np.tile([3, -3], 5),
                     index=pd.date_range("2026-08-24", periods=10, freq="7D"))
    fc = forecast_weekly(hist, horizon=8, uplift_pct=50)
    hol = fc[fc.is_holiday]
    assert len(hol) == 4
    assert np.allclose(hol.forecast, hol.baseline * 1.5)
    assert (fc[~fc.is_holiday].forecast == fc[~fc.is_holiday].baseline).all()
    width = (fc.high - fc.low) / np.where(fc.is_holiday, 1.5, 1.0)
    assert width.is_monotonic_increasing


def test_capacity_gaps():
    fc = pd.DataFrame({"forecast": [90, 110, 95], "high": [95, 130, 105], "low": [80, 90, 85]})
    g = capacity_gaps(fc, 100)
    assert g["over"].tolist() == [False, True, False]
    assert g["at_risk"].tolist() == [False, False, True]
