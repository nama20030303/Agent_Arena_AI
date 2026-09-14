# Overfitting, Regularization and the Bias–Variance Trade-off

*ML Engineer Academy sample material — self-authored. Chapter 6.*

## 6.1 The error decomposition

For squared loss, the expected error of a learned model at a point decomposes into three terms:

```text
E[(f(x) - ŷ(x))²] = (E[ŷ(x)] - f*(x))²  +  E[(ŷ(x) - E[ŷ(x)])²]  +  σ²
                     \___ bias² ______/    \______ variance ______/   \_ noise
```

- **Bias**: the systematic error of the average predictor the procedure produces. High bias means the
  hypothesis class cannot represent the truth (a line fitted to a curve).
- **Variance**: how much the prediction moves when you retrain on a different sample of the same size.
  High variance means the procedure is hypersensitive to the particular data you happened to get.
- **Noise**: irreducible; no model can beat it, so a "best possible" score is not 0.

The trade-off is a property of the *procedure + capacity + data size*, not of a single trained model.
Adding data reduces variance without increasing bias. Adding capacity reduces bias and increases variance.

## 6.2 Recognising each regime from the learning curves

Plot training and validation error against number of epochs and against dataset size:

| Curves | Reading | Action |
|---|---|---|
| both high and close together | underfitting (bias) | more capacity, better features, less regularization |
| train low, validation high, gap wide | overfitting (variance) | more data, regularization, early stopping, simpler model |
| validation dips then rises | overfitting in time | early stopping at the dip |
| both high and far apart on time-series | temporal shift | rebuild the split; check leakage |
| train ≈ validation but both worse than a baseline | wrong inductive bias | change the model family, not the hyperparameters |

The "U-shape" of validation error vs capacity is the textbook signal; in deep learning the second
descent (validation improving again after the overfitting peak) is real and mostly attributed to the
optimiser preferring flat minima.

## 6.3 Regularization is a prior, not a magic knob

Adding λ‖w‖₂² to the loss is identical to placing a zero-mean Gaussian prior on the weights and doing
MAP estimation. L1 corresponds to a Laplace prior and produces exact zeros because the penalty's
contour has corners on the axes; the sub-gradient contains an interval at zero, so weights get pinned there.

```text
ridge:  w* = (XᵀX + λI)⁻¹ Xᵀy      -> shrinks all coefficients, keeps them non-zero
lasso:  minimize ½‖Xw - y‖² + λ‖w‖₁ -> sparsity, feature selection, unstable with correlated features
elastic net: both terms -> keeps groups of correlated features instead of picking one at random
```

Practical notes that are often missed:
- Regularization strength is only comparable **after** feature scaling. λ on unscaled features
  penalizes small-unit variables unfairly.
- The intercept/bias is conventionally not penalized; in a matrix implementation that means a zero
  in the diagonal of the penalty matrix.
- With ridge, one fit lets you sweep all λ via the SVD: recompute nothing but the shrinkage factors
  `s²/(s²+λ)` — an O(1) per-λ evaluation after an O(nd²) decomposition.
- In neural nets, weight decay ≠ L2 inside the gradient when using Adam (see AdamW); dropout is a
  different mechanism (approximate ensemble averaging), not a rescaling trick.

## 6.4 Data augmentation and noise injection as regularizers

Augmentation encodes an invariance you believe in. Flipping images for a "is this a car" task is
legitimate; flipping text is not. Adding Gaussian noise to inputs is equivalent (to second order) to
Tikhonov regularization; label smoothing bounds the confidence and typically improves calibration
while slightly hurting accuracy — which is often the right trade for a decision system.

## 6.5 Model selection without fooling yourself

The evaluation protocol is part of the model:

1. **train** to fit weights,
2. **validation** to choose hyperparameters/early-stop,
3. **test** once, at the end, and never to make a decision.

If you use the test set to pick a config, it silently became the validation set: report the number
only if you can also report how many times you looked at it.

Stratify on the label; group-split when samples share an entity (patient, user, session, image-of-the-same-object);
time-split for anything with a temporal component — and never mix them into one metric table.

## 6.6 Common interview follow-ups (and the honest answers)

- *“More data always helps?”* — helps variance; useless if you are bias-bound. Check by learning curve.
- *“Dropout or batch norm?”* — different axes: dropout regularizes, BN conditions the optimisation
  (and has a small regularizing side-effect through batch noise). Both can be removed if you
  have enough data and a well-tuned schedule.
- *“Early stopping is regularization?”* — yes: it constrains the distance travelled from
  initialisation, which is similar in effect to weight decay (a small-norm solution).
- *“Your validation metric improved 0.3% — ship it?”* — no. Estimate the noise on the metric
  (bootstrap or repeated CV); if the CI crosses zero you have not measured anything.

## 6.7 Practice prompts

1. Derive the ridge closed form from the penalised objective, then show why it never produces exact zeros.
2. Implement ridge with gradient descent on standardised data, sweep λ ∈ {1e-3 … 1e3}, and plot the
   coefficient path (this is the "ridge trace" — describe what you see for correlated features).
3. Construct a dataset where a model has bias ≈ 0 and huge variance; explain how you detected it.
4. Your random forest overfits (train 1.0, OOB 0.72). List four changes, ranked by expected effect size.
