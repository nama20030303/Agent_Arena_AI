# Linear and Logistic Regression: The Models You Will Re-Implement Forever

*ML Engineer Academy sample material — self-authored. Chapter 2.*

## 2.1 Linear regression as a projection

We want the vector of predictions `ŷ = Xw` to be as close as possible to `y`. Least squares
minimises `||Xw - y||²`, whose solution is the orthogonal projection of `y` onto the column
space of `X`. Setting the gradient to zero gives the normal equations:

```text
Xᵀ(Xw - y) = 0   =>   w = (XᵀX)⁻¹ Xᵀ y
```

Practically, never invert `XᵀX`: use `numpy.linalg.lstsq` or a QR/SVD solve. `XᵀX` squares the
condition number, so with correlated features the inverse is numerically unstable while
`lstsq` is not.

Diagnosis checks worth memorising:
- residual mean must be ~0 and residuals must not correlate with any feature;
- residual variance should be roughly constant across the fitted range (heteroscedasticity
  means the model's uncertainty is wrong even when the mean is right);
- `R²` on the test set being much lower than on train is overfitting, not "bad luck".

## 2.2 Regularisation is a prior, not a trick

Adding `λ||w||²` (ridge) is equivalent to a zero-mean Gaussian prior on the weights; the
closed form becomes `w = (XᵀX + λI)⁻¹Xᵀy`. The added diagonal makes the system solvable even
when `XᵀX` is singular (perfectly collinear columns) — that alone explains why ridge is a
safer default than unregularised least squares.

Lasso (`λ||w||₁`) has the subgradient form, which can pin a coefficient exactly at zero:
sparsity as an optimisation property, not a post-hoc selection step.

Scaling note: a penalty applied to unstandardised features punishes small-unit features more
heavily. If you skip scaling, `λ` becomes meaningless, which is the most common bug in a
first notebook.

## 2.3 Logistic regression: modelling log-odds

```text
p = σ(w·x + b) = 1 / (1 + e^-(w·x + b))
log-odds = log(p / (1 - p)) = w·x + b
L(w) = -1/n Σ [ yᵢ log pᵢ + (1 - yᵢ) log(1 - pᵢ) ]      (negative log-likelihood)
∂L/∂w = 1/n Xᵀ(p - y)
```

Read coefficients on the log-odds scale: `exp(wⱼ)` is the odds ratio for a one-unit increase in
feature `j`. Saying "the probability increases by 0.3" is the classic error — log-odds are
additive, probabilities are not.

The gradient `(p - y)` is beautiful: it is exactly "how wrong are you, times which feature".
When `p ≈ y` for all rows, the model has nothing left to learn. This is also why MSE on a
sigmoid is a poor objective: it adds a factor `σ'` that vanishes precisely where the model is
most confident and most wrong.

## 2.4 Softmax for multi-class

For `K` classes, `p = softmax(Wx)`, and the cross-entropy gradient w.r.t. the logits is
`(p - onehot(y)) / n`. Same shape as logistic regression; that is not a coincidence — binary
logistic regression is a two-class softmax with one redundant column removed.

Softmax is shift-invariant, which is the practical trick that avoids overflow:
`softmax(z) = softmax(z - max(z))`.

## 2.5 Calibration: the part that decides whether the model is usable

A model can rank perfectly and still be useless for decisions if its probabilities are wrong.
Linear regression has no notion of probability; logistic regression is properly calibrated only
if the link function is correct and the model is not overfit (regularised logistic regression is
naturally *under*-confident, overfit ones are *over*-confident).

Tooling: reliability diagram (bucket predictions, compare mean predicted vs observed frequency),
Brier score, then a recalibration map — Platt scaling (fit a 1-D logistic on the scores) or
isotonic regression (non-parametric, needs more data, cannot be applied out of the training
range).

Decision rule: if you will threshold the score, calibrate *after* the threshold search, on a
held-out set, or you will ship a threshold that only exists in the tuning data.

## 2.6 From model to decision

The threshold is not a hyperparameter of the model, it is a contract with the business:

```text
expected_cost(threshold) = FP(threshold) * cost_fp + FN(threshold) * cost_fn
```

Sweep it, report precision/recall/volume at the chosen point, and monitor the score
distribution online: a shift in the score histogram at fixed threshold silently changes your
operating point. This is why production systems ship *thresholds as configuration*, not
constants in code.

## 2.7 Exercises

1. Derive the ridge gradient and show the closed form.
2. Implement linear regression with full-batch gradient descent on standardised data and compare
   with `numpy.linalg.lstsq` to 1e-6.
3. Show that with two perfectly collinear features the unregularised fit is undefined but ridge
   is not; plot the coefficient path over `λ ∈ [1e-4, 1e2]`.
4. Overfit a logistic regression on 200 samples with 50 features; measure the reliability
   diagram before and after `C=0.05`. What changed: ranking or calibration, or both?
5. Given `y_true` and scores, compute the threshold maximising F1 under the constraint
   precision ≥ 0.6, and report the resulting alert volume per day at 10M events/day.
