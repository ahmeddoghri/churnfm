"""Synthetic churn dataset generator with a built-in concept drift.

Simulates a B2B subscription business: usage, support tickets, tenure, and
plan tier predict churn. Halfway through the stream the *true* relationship
shifts (a pricing change makes price-sensitivity matter much more) — the kind
of concept drift that silently rots a churn model in production and is exactly
why automated retraining triggers matter.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Row:
    tenure_months: float
    monthly_usage: float
    support_tickets: int
    plan_tier: int          # 0=basic, 1=pro, 2=enterprise
    price_increase_pct: float
    churned: int


def _logistic(x: float) -> float:
    import math
    return 1 / (1 + math.exp(-x))


def generate(n: int = 2000, seed: int = 0, drift_at: float = 0.5) -> list[Row]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        post_drift = (i / n) >= drift_at
        tenure = rng.uniform(0, 48)
        usage = max(0.0, rng.gauss(50, 20))
        tickets = max(0, int(rng.gauss(2, 2)))
        tier = rng.choice([0, 1, 2])
        price_increase = rng.uniform(0, 30) if post_drift else rng.uniform(0, 5)

        # true generative process
        z = -1.5
        z += -0.03 * tenure            # loyalty reduces churn
        z += -0.02 * usage             # engaged users churn less
        z += 0.15 * tickets            # friction increases churn
        z += -0.3 * tier                # higher tier -> stickier
        if post_drift:
            z += 0.35 * price_increase  # NEW after drift: price sensitivity dominates
        else:
            z += 0.01 * price_increase  # negligible before drift
        p = _logistic(z)
        churned = 1 if rng.random() < p else 0
        rows.append(Row(tenure, usage, tickets, tier, price_increase, churned))
    return rows


def features(row: Row) -> list[float]:
    return [row.tenure_months, row.monthly_usage, float(row.support_tickets),
            float(row.plan_tier), row.price_increase_pct]


FEATURE_NAMES = ["tenure_months", "monthly_usage", "support_tickets",
                 "plan_tier", "price_increase_pct"]
