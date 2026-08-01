"""Drift scenarios that isolate what actually changed.

The bundled benchmark injects "drift" by changing three things at once at the
stream midpoint:

1. the coefficient on ``price_increase_pct`` (0.01 -> 0.35) -- real concept drift
2. the *distribution* of ``price_increase_pct`` (0-5 -> 0-30) -- covariate shift
3. the churn base rate, as a side effect of both (3.2% -> 68.4%)

That third change is what actually explains the benchmark's headline number.
A 68% base rate makes churn nearly a coin a static model can call correctly
most of the time just by leaning toward "churned", which is why the *static*
model's post-drift precision is 80%, not some collapsed number. The scenario
was accidentally easy after the drift, not just different.

``pure_concept_drift`` isolates the thing PSI is supposed to detect: the
relationship between features and outcome inverts, while the feature
distribution and the churn base rate are held approximately constant by
construction. Nothing about the *inputs* changes, only what they mean. This is
the hard case: a static model's predictions look plausible, the input
distribution looks stable, and only the correspondence between the two has
rotted.

``covariate_shift_confound`` is the original scenario, kept and named for what
it actually is, so the difference is visible in the benchmark rather than
silently baked into one "drift" scenario with no alternative to compare against.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .data import Row


def _logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def covariate_shift_confound(n: int = 2000, seed: int = 0, drift_at: float = 0.5) -> List[Row]:
    """The original scenario: coefficient, input distribution, and base rate
    all change together. Kept for comparison against the isolated scenario
    below; the aggregate name says what it actually tests.
    """
    rng = random.Random(seed)
    rows: List[Row] = []
    for i in range(n):
        post_drift = (i / n) >= drift_at
        tenure = rng.uniform(0, 48)
        usage = max(0.0, rng.gauss(50, 20))
        tickets = max(0, int(rng.gauss(2, 2)))
        tier = rng.choice([0, 1, 2])
        price_increase = rng.uniform(0, 30) if post_drift else rng.uniform(0, 5)

        z = -1.5 - 0.03 * tenure - 0.02 * usage + 0.15 * tickets - 0.3 * tier
        z += 0.35 * price_increase if post_drift else 0.01 * price_increase
        p = _logistic(z)
        rows.append(Row(tenure, usage, tickets, tier, price_increase,
                        1 if rng.random() < p else 0))
    return rows


def pure_concept_drift(n: int = 3000, seed: int = 0, drift_at: float = 0.5) -> List[Row]:
    """The relationship inverts; the inputs and base rate do not move.

    Every feature is drawn from the same distribution before and after the
    drift point. Only the sign and weight of each feature's effect on churn
    changes: engaged, high-usage, long-tenure customers were safe before and
    are now the ones leaving (a real pattern when, say, a product pivot
    alienates power users while attracting a short-term influx of casual
    ones). Coefficients are centered on each feature's typical value so the
    swap does not itself shift the average predicted probability, which is
    what keeps the base rate matched by construction rather than by luck.
    """
    rng = random.Random(seed)
    rows: List[Row] = []
    for i in range(n):
        post_drift = (i / n) >= drift_at
        tenure = rng.uniform(0, 48)
        usage = max(0.0, rng.gauss(50, 20))
        tickets = max(0, int(rng.gauss(2, 2)))
        tier = rng.choice([0, 1, 2])
        price_increase = rng.uniform(0, 30)  # unchanged across the drift point

        if post_drift:
            z = -1.0 + 0.04 * (usage - 50) - 0.15 * (tickets - 2) + 0.02 * (tenure - 24)
        else:
            z = -1.0 - 0.04 * (usage - 50) + 0.15 * (tickets - 2) - 0.02 * (tenure - 24)
        p = _logistic(z)
        rows.append(Row(tenure, usage, tickets, tier, price_increase,
                        1 if rng.random() < p else 0))
    return rows


def holdout_concept_drift(n: int = 3600, seed: int = 99, drift_at: float = 0.5) -> List[Row]:
    """A second, differently-shaped concept drift, written after the outcome
    drift threshold was frozen and evaluated exactly once. Tenure and support
    tickets swap which one protects against churn, rather than usage and
    tenure as in :func:`pure_concept_drift`. Different generative shape, same
    isolation: base rate and inputs held constant, only the relationship
    inverts.
    """
    rng = random.Random(seed)
    rows: List[Row] = []
    for i in range(n):
        post_drift = (i / n) >= drift_at
        tenure = rng.uniform(0, 60)
        usage = max(0.0, rng.gauss(40, 15))
        tickets = max(0, int(rng.gauss(3, 2)))
        tier = rng.choice([0, 1, 2])
        price_increase = rng.uniform(0, 20)

        if post_drift:
            z = -0.8 - 0.10 * (tickets - 3) + 0.03 * (tenure - 30) - 0.01 * (usage - 40)
        else:
            z = -0.8 + 0.10 * (tickets - 3) - 0.03 * (tenure - 30) + 0.01 * (usage - 40)
        p = _logistic(z)
        rows.append(Row(tenure, usage, tickets, tier, price_increase,
                        1 if rng.random() < p else 0))
    return rows


@dataclass(frozen=True)
class ScenarioCheck:
    name: str
    pre_base_rate: float
    post_base_rate: float
    #: How far the two windows' input feature means diverge, as a fraction.
    #: Near zero confirms the drift is in the relationship, not the inputs.
    feature_shift: float


def describe(rows: List[Row], drift_at: float = 0.5, name: str = "") -> ScenarioCheck:
    """Report what actually moved across the drift point, for verification."""
    split = int(len(rows) * drift_at)
    pre, post = rows[:split], rows[split:]
    pre_rate = sum(r.churned for r in pre) / len(pre)
    post_rate = sum(r.churned for r in post) / len(post)

    def mean_usage(subset: List[Row]) -> float:
        return sum(r.monthly_usage for r in subset) / len(subset)

    pre_usage, post_usage = mean_usage(pre), mean_usage(post)
    shift = abs(post_usage - pre_usage) / (pre_usage or 1.0)

    return ScenarioCheck(name, round(pre_rate, 4), round(post_rate, 4), round(shift, 4))
