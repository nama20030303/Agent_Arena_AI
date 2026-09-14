# Bias–Variance, Evaluation Honesty and the Learning Curve

*ML Engineer Academy sample material — self-authored. Chapter 3.*

## 3.1 The decomposition, without the hand-waving

For a squared-error problem, the expected error at a point decomposes as:

```text
E[(f(x) - ŷ(x))²] = (Bias[ŷ(x)])² + Var[ŷ(x)] + σ²
```

`σ²` is irreducible noise. Bias is the systematic error of the *average* predictor over
training sets; variance is how much the predictor moves when you resample the training data.
The trade-off is not "simple vs complex" in the aesthetic sense — it is a constraint: you can
reduce one term only by increasing the other, unless you add data or better features.

Diagnosis table that is worth memorising:

| Symptom | Interpretation | First three actions |
|---|---|---|
| train ≈ test, both bad | high bias / underfitting | more capacity, better features, less regularisation |
| train good, test bad | high variance / overfitting | more data, regularisation, simpler model, early stopping |
| train good, test good, online bad | distribution shift or leakage | compare feature stats online vs training window |
| test improves then degrades with epochs | optimisation overfitting | early stopping at the validation minimum |

## 3.2 Learning curves decide which of the two you have

Plot error against training-set size (holding the evaluation set fixed):

- both curves converge to a high error → variance is not the problem, adding data will not help;
  you need a different hypothesis class;
- a wide gap that shrinks slowly → variance dominates, and more data helps; estimate the
  marginal benefit before buying/labeling it;
- a plateau at high train error with tiny data → normal; extend the curve before concluding.

## 3.3 Leakage: the top cause of a dead ML project

Leakage is any information at training time that will not be available at prediction time.

Common, easy-to-miss forms:
1. **Target-derived features**: "days since cancellation" in a churn model, aggregates that
   include the label window.
2. **Time-order violations**: random row splits on event data; the model memorises the future.
   Split by an observation cut-off, not by shuffled rows.
3. **Group leakage**: the same user/account/device in train and test. Use group-aware splits.
4. **Pre-fit transformers**: scaler/imputer/encoder fit on all data.
5. **Duplicate rows** across the split boundary (same event with different timestamps).
6. **Post-hoc features** produced by an offline pipeline that ran *after* the event
   (e.g. a enrichment service whose refresh is daily).

The structural defence is point-in-time correctness: for every row, the features you use must
be reconstructable from data that existed at the event timestamp. Feature stores exist largely
to make this checkable rather than tribal knowledge.

## 3.4 Choosing the metric backwards, from the decision

Write the decision before the model:
```text
action = predict(score > threshold)
expected_value(threshold) = Σ outcome(score, label) * probability
```
Then:
1. is the metric of interest a ranking property (ROC AUC, PR AUC, nDCG) or a probability
   property (log-loss, Brier, calibration)?
2. what does an FP cost vs an FN? The ratio defines the operating threshold, not the model.
3. how much data is in the tail you care about? Recall on 0.2% positives has enormous
   confidence intervals; report them.

Report the *set* {precision, recall, volume at threshold, cost per unit time}, never a lone
number, and always against a baseline (majority class, rule-based, previous model).

## 3.5 Cross-validation is a variance-reduction device

- `k` bigger → less bias in the estimate, more variance in fold-to-fold differences, more compute.
- Stratify on the label; group-split on the entity; time-split on the timestamp. If you can only
  do one, do the time split for anything with a temporal component.
- Report mean and std across folds, and never pick the model with the best fold mean without a
  nested comparison — selection on noisy estimates is itself overfitting (the "winner's curse").
- Repeated CV (multiple random seeds) is cheap insurance when folds < 10.

## 3.6 Statistical honesty in the report

1. One number is not a result: bootstrap or use a paired test between models on the same folds.
2. Compare models on identical splits — different folds invalidate every comparison.
3. State the evaluation protocol before the numbers, in the same document.
4. Distinguish "better offline" from "better" — the second requires an online experiment.

## 3.7 Exercises

1. Draw the two learning-curve shapes for bias- and variance-dominated regimes and label the
   asymptotes.
2. You are given `train (800k rows) / test (200k)` on 18 months of transactions. Design the split
   so that it is temporally honest and has no user overlap. Write the exact code.
3. A gradient-boosted model: log-loss 0.18, accuracy 97.2%, on a 1% positive class. Explain what
   each number hides and which measurement would actually decide whether to ship.
4. Show mathematically why adding training data cannot reduce squared bias.
5. Find (and fix) the leakage in this snippet: `X['hist_mean'] = df.groupby('user')['amt'].transform('mean')`
   used to predict whether the *same* transaction is fraudulent.
