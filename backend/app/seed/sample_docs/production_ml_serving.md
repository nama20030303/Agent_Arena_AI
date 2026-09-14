# Serving a Model: From Notebook to P99 Latency

*ML Engineer Academy sample material — self-authored. Chapter 5.*

## 5.1 The three contracts of an ML service

1. **Input contract**: the exact schema, units and missingness policy of every feature.
   A model does not fail when data is wrong; it fails silently when data is *different*.
2. **Output contract**: calibrated probability vs score vs ranking; the consumer's threshold
   depends on which one you promise.
3. **Behaviour contract**: latency budget, degradation ladder, and what happens when the model
   is unavailable (fallback score, cached result, or conservative default).

Write all three down as types (`pydantic` / `pandera` / protobuf), because the only contract that
survives a year of team turnover is the one the machine checks.

## 5.2 Latency anatomy

```text
T_total = T_queue + T_feature_fetch + T_preprocess + T_inference + T_postprocess + T_network
```

Measure each term separately; the dominant one is rarely what people guess:

- GPU inference is often *not* the bottleneck for small models — feature fetch from a remote
  store is (many round trips instead of one batched read).
- Python preprocessing can exceed model time when it uses pandas per request. Move it to NumPy
  on pre-shaped buffers, or push it to the feature store.
- Queueing: at 85% average utilisation, p99 explodes (waiting time grows hyperbolically). Capacity
  planning must target the *tail*, not the mean.

Techniques that reliably help: request batching with a micro-timeout (e.g. 5 ms), pinned warm
model, `torch.inference_mode()`, half precision when validated, avoiding per-request allocations,
and caching immutable feature groups (user profile) with an explicit invalidation policy.

## 5.3 Failure design: the degradation ladder

```text
1. full model
2. model with reduced feature set (features known to be stale dropped)
3. previous known-good model version
4. cheap statistical baseline (popularity / prior probability)
5. conservative no-op (human review, or reject-with-retry)
```

Each rung must be *reachable by configuration or automatically*, and each transition must emit a
metric. If the fallback is not tested, you will discover at 3 a.m. that it was broken since March.

## 5.4 What to monitor (and how to alert without crying wolf)

| Signal | Measure | Alert condition |
|---|---|---|
| Feature health | null rate, type violations, PSI vs training window | sustained shift, not a single spike |
| Score distribution | mean/quantiles, positive rate | drift beyond a control band for N windows |
| Quality | delayed-label metrics with confidence intervals | lower bound below the acceptance bar |
| Ops | p50/p99 latency, error rate, queue depth, GPU util | SLI/SLO error-budget burn rate |
| Economics | cost per 1k predictions, decisions per day | budget burn, not absolute cost |

Two habits separate mature teams: (a) alerts are owned, with a runbook link; (b) the label-delay
window is explicitly modelled, so a quality dashboard says "as of data available up to T".

## 5.5 Rollouts

Shadow → canary (1% → 10% → 50%) → full, with automatic rollback on guardrail breach (error rate,
latency, positive-rate jump). Keep the incumbent warm during the canary so rollback is a config
flip, not a deployment. Record every promotion decision in the model registry: which evaluation
set, which numbers, who approved, and which commit/image/data version produced the artefact.

## 5.6 Cost per prediction

```text
cost_per_1k = (instances * hourly_cost / throughput_per_hour) * 1000
```
Break it down and rank the levers by `saved_cost / engineering_weeks`. Typical order for a CPU
service: quantise the model, batch requests, cache immutable features, right-size instances,
then and only then consider a smaller architecture. Every lever must be re-measured against the
offline evaluation set — a 40% cost saving that costs 2 points of AUC at the operating threshold
is usually a bad trade, and the sentence before this one is the one to say out loud in review.

## 5.7 Exercises

1. Your p99 is 340 ms with a 120 ms budget. Write the measurement plan that finds the term
   responsible before you change any code.
2. Design the schema validation that rejects `{"age": "unknown"}` at the edge and not inside the model.
3. Show the arithmetic for capacity at 400 RPS with a mean of 22 ms of CPU work per request,
   targeting a 35% headroom.
4. Which of the five rungs in 5.3 is missing from your current service, and how would you test it?
