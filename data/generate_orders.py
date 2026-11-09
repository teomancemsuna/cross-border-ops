"""
Generate a reproducible synthetic dataset of cross-border marketplace orders
(US warehouse -> Canadian customers), 1 May - 1 Nov 2026.

Every order ID, product and figure is invented. The random seed is fixed, so
running this script always produces exactly the same data/orders.csv.

Run:  python data/generate_orders.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-11-01")
PROMISE_DAYS = 12          # delivery date promised to the customer at checkout
LOST_RATE = 0.012          # share of parcels that never get delivered
SNAPSHOT = pd.Timestamp("2026-11-02")
RESOLVED_AFTER_DAYS = 45   # lost parcels older than this were already refunded

# Shipping routes: cross-border carrier -> Canadian last-mile carrier.
# transit = (mean, std) days from warehouse hand-off to delivery.
ROUTES = {
    "Asendia → Canada Post": dict(share=0.50, transit=(7.0, 1.6), cost_usd=(9.5, 1.2)),
    "Purolator (end-to-end)": dict(share=0.30, transit=(5.4, 1.4), cost_usd=(14.5, 1.5)),
    "Asendia → Intelcom": dict(share=0.20, transit=(6.4, 2.6), cost_usd=(8.2, 1.0)),
}

# Province share and extra last-mile days for remote regions
PROVINCES = {
    "ON": (0.39, 0.0), "QC": (0.22, 0.3), "BC": (0.14, 1.0), "AB": (0.12, 0.6),
    "MB": (0.04, 0.8), "SK": (0.03, 0.9), "NS": (0.04, 1.4), "NL": (0.02, 2.2),
}

CATEGORIES = {
    "Home & Kitchen": (0.35, 38.0), "Toys & Games": (0.22, 32.0),
    "Beauty": (0.18, 26.0), "Electronics Accessories": (0.15, 44.0),
    "Sports & Outdoors": (0.10, 52.0),
}


def build_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    days = pd.date_range(START, END, freq="D")

    # Daily order volume: gentle growth, a weekly pattern and Poisson noise.
    t = np.arange(len(days))
    weekday_factor = np.array([1.05, 1.0, 1.0, 1.02, 1.08, 0.9, 0.85])[days.dayofweek]
    expected = (24 + 0.06 * t) * weekday_factor
    counts = rng.poisson(expected)

    order_date = np.repeat(days.values, counts)
    n = len(order_date)

    route_names = list(ROUTES)
    route = rng.choice(route_names, size=n, p=[ROUTES[r]["share"] for r in route_names])
    prov_names = list(PROVINCES)
    province = rng.choice(prov_names, size=n, p=[PROVINCES[p][0] for p in prov_names])
    cat_names = list(CATEGORIES)
    category = rng.choice(cat_names, size=n, p=[CATEGORIES[c][0] for c in cat_names])

    # Warehouse prep: 1-2 days, slower on Fridays/weekends (picked up Monday).
    order_ts = pd.to_datetime(order_date)
    prep = rng.integers(1, 3, size=n) + np.where(order_ts.dayofweek >= 4, 1, 0)

    transit_mean = np.array([ROUTES[r]["transit"][0] for r in route])
    transit_std = np.array([ROUTES[r]["transit"][1] for r in route])
    remote = np.array([PROVINCES[p][1] for p in province])
    transit = np.maximum(2, np.round(rng.normal(transit_mean + remote, transit_std)))

    ship_date = order_ts + pd.to_timedelta(prep, unit="D")
    delivered = ship_date + pd.to_timedelta(transit, unit="D")
    lost = rng.random(n) < LOST_RATE
    delivered = delivered.where(~lost, pd.NaT)

    cost_mean = np.array([ROUTES[r]["cost_usd"][0] for r in route])
    cost_std = np.array([ROUTES[r]["cost_usd"][1] for r in route])
    price_mean = np.array([CATEGORIES[c][1] for c in category])

    df = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
        "order_date": order_ts,
        "category": category,
        "province": province,
        "route": route,
        "ship_date": ship_date,
        "promised_date": order_ts + pd.Timedelta(days=PROMISE_DAYS),
        "delivered_date": delivered,
        "sale_price_cad": np.round(np.maximum(9.99, rng.normal(price_mean, price_mean * 0.25)), 2),
        "shipping_cost_usd": np.round(np.maximum(5.0, rng.normal(cost_mean, cost_std)), 2),
        # Older lost parcels have already been handled (customer refunded).
        "refunded": lost & ((SNAPSHOT - order_ts).days >= RESOLVED_AFTER_DAYS),
    })
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "orders.csv"
    df = build_dataframe()
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(df):,} orders to {out}")
