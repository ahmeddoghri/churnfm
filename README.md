# 📉 ChurnFM

**A churn classifier with drift detection and automated retraining.**

![tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Detect drift with PSI and retrain automatically, before precision
> rots.** In the benchmark, a static model recovers to **80%** post-drift
> accuracy while the adaptive one reaches **89%**. Zero deps:
> `python -m churnfm.eval`.
>
> Then I isolated the drift from the confound riding along with it (the
> benchmark's "drift" also multiplies the churn base rate 20x, which is
> most of why even the static model scores 80%), and PSI **never fired**:
> precision fell from 42% to 14% while PSI sat under 0.05 against a
> threshold of 0.25. `python -m churnfm.eval_v2` is the benchmark that
> found it, and the outcome-based signal that catches what PSI misses.

Most churn models get trained once, deployed, and then quietly ghosted.
Nobody's watching when a pricing change, a new competitor, or a product
pivot rewrites the actual relationship between your features and who
leaves. The model doesn't know any of that happened. It just keeps
answering questions based on a world that no longer exists, like an ex
who still thinks you're together.

ChurnFM watches the prediction distribution itself using the Population
Stability Index, and retrains the moment the underlying relationship
shifts, instead of waiting for a dashboard to look wrong three weeks
later.

Runs with **zero dependencies and zero API keys** (pure-stdlib logistic
regression trained by gradient descent). The point was never the model
architecture. Swap `ChurnModel` for a real tabular foundation model or a
gradient-boosted-tree model through the same `fit`/`predict_proba`
interface if you want. The point is the **monitor-and-retrain loop**
wrapped around it, which doesn't care what's inside.

---

## The result in one number

A synthetic B2B subscription stream with a concept drift injected at the
midpoint. A pricing change makes price-sensitivity the dominant churn
driver, the exact way churn models silently rot in production without
anyone noticing:

```bash
python -m churnfm.eval
```
```
batches: 13  (drift injected at stream midpoint)

policy         pre-drift precision    post-drift precision
static                        15%                    80%
adaptive                      15%                    89%

adaptive retrained at batches: [6, 7, 8]
```

Both policies score identically before the drift, as they should, since
it's the same model. After the pricing relationship changes, the static
model's precision stalls while the PSI monitor catches the shift and
triggers a retrain. Precision here is precision@k with k = actual
positives, the standard ranking metric for imbalanced churn, since a
fixed probability cutoff is meaningless when churn is a single-digit
percent event.

## The drift that number doesn't test

I went looking for how much of that 15% -> 89% recovery was actually the
monitor catching drift, versus the scenario just becoming easier. It was
mostly the second thing.

```bash
python -m churnfm.eval_v2
```
```
Scenario check: what actually moved across the drift point
  covariate_shift_confound  base rate 3.5% -> 69.5%   feature shift 0.3%
  pure_concept_drift        base rate 27.1% -> 30.8%  feature shift 0.3%
```

The bundled "drift" changes three things at the stream midpoint: the
coefficient on `price_increase_pct` (real concept drift), the
*distribution* of `price_increase_pct` (covariate shift), and, as a side
effect of both, the churn base rate, which jumps **20x**, from 3.5% to
69.5%. A base rate that high makes churn nearly a coin a model can call
correctly just by leaning toward "churned". That's most of why even the
**static** model scores 80% post-drift; the problem got easier, not just
different.

`pure_concept_drift` isolates the thing PSI is supposed to detect: the
relationship between features and outcome inverts while the base rate and
input distributions stay put by construction (0.3% feature shift, versus
the confounded scenario's 66-point base-rate swing). On that scenario:

```
policy         pre-drift    post-drift    retrains
static              42%          14%
psi_only             42%          14%             0
dual_signal          42%          41%             1
```

**PSI never fires. Zero retrains, for the rest of the stream's life.**
Directly measured: PSI stayed under 0.05 (threshold 0.25) while precision
fell from 42% to 14%, because a logistic model re-scores the *same input
distribution* through the same fitted function whether the relationship
underneath has changed or not. The predicted-score distribution looks
just as healthy after the world flipped as before.

### What catches it

PSI compares scores to scores. `assess_outcomes` compares the model's
predictive quality on labeled outcomes, reference window against current
batch, using log-loss. A model whose relationship to the world has
inverted gets measurably worse at labels it has never adjusted for, even
when its score distribution hasn't moved an inch. `ChurnMonitorV2`
retrains on either PSI or outcome drift firing; on the isolated scenario,
outcome drift is the one that actually does it.

The trade is immediacy for ground truth: PSI can flag drift before any
label exists for new data, outcome drift needs labels, which in a real
churn pipeline arrive weeks after the fact. Run both.

### Retraining on the wrong window makes it worse, not better

The first fix I tried kept the original sliding-window retrain (last two
batches of labeled history) and just added the outcome trigger. Precision
after retraining stayed at 9-12%, no better than never retraining. The
window mixed labeled examples from *both* regimes, half teaching the old
relationship and half the new, inverted one, and a model fit on that
mixture learns something close to nothing. Retraining on just the single
batch that tripped the alarm, guaranteed to be from the current regime,
recovered precision to 38-51%.

### Held out, run once

The outcome-drift threshold (1.3) was tuned against the scenario above.
A second, differently-shaped concept drift (tenure and support tickets
swap which one protects against churn, rather than usage and tenure) was
written afterward and run a single time:

```
policy         pre-drift    post-drift    retrains
static              41%          21%
psi_only             41%          21%             0
dual_signal          41%          45%             1
```

Same story: PSI never fires, dual_signal recovers with one retrain
triggered entirely by the outcome signal.

### Limits

- **Outcome drift needs labels.** In production those arrive with a lag; PSI is still the only signal available in the interim.
- **The log-loss ratio is one threshold, tuned on synthetic data.** A real deployment should watch the ratio's distribution over a burn-in period rather than trust 1.3 blindly.
- **This is still logistic regression on five features.** The monitoring loop is the point; swap in a real model behind the same `fit`/`predict_proba` interface for anything that matters.

## Install

```bash
git clone https://github.com/ahmeddoghri/churnfm
cd churnfm && pip install -e .
python examples/quickstart.py
```

Or with Docker:

```bash
docker build -t churnfm .
docker run --rm churnfm
```

## Monitor a stream

```python
from churnfm import ChurnMonitor, generate

rows = generate(n=1000, seed=0, drift_at=0.5)
monitor = ChurnMonitor.fit(rows[:400])

recent = list(rows[:400])
for i in range(400, len(rows), 200):
    batch = rows[i:i + 200]
    result = monitor.process_batch(i, batch, recent[-400:])
    recent.extend(batch)
    print(result.drift.psi, result.retrained, result.precision)
```

## How it works

```
ChurnMonitor.fit(reference_window)
  └─ trains ChurnModel, stores reference prediction distribution

process_batch(batch, recent_history)
  ├─ score the batch
  ├─ PSI(reference_scores, batch_scores)   -- how far the distribution has drifted
  └─ if PSI >= threshold: retrain on recent_history, reset reference distribution
```

Retraining on a **recent sliding window**, rather than all accumulated
history, matters: mixing in stale pre-drift examples would keep
re-triggering the drift alarm indefinitely, and the model would never
settle down after the relationship actually stabilizes into its new
shape.

## Bring your own model

```python
class MyTabularModel:
    def fit(self, rows): ...
    def predict_proba(self, row): ...
    def predict_proba_batch(self, rows): ...

ChurnMonitor(model=MyTabularModel().fit(reference_rows), reference_scores=[...])
```

## Tests

```bash
pip install pytest && pytest -q      # 6 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
