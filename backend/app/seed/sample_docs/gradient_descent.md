# Gradient Descent: From Derivatives to Training Loops

*ML Engineer Academy sample material — self-authored, free to copy. Chapter 1.*

## 1.1 What the algorithm is trying to do

Every supervised model has an objective: a number we want small. Gradient descent is the
cheapest general-purpose way to reduce it. We start at some parameters `w`, ask "in which
direction does the loss rise fastest?", and move the opposite way by a small amount.

Formally, for a differentiable loss `L(w)`:

```text
w_{t+1} = w_t - lr * grad L(w_t)
```

The gradient is the vector of partial derivatives. Because the first-order Taylor
expansion says `L(w - t*g) ≈ L(w) - t * ||g||^2`, the loss decreases for sufficiently
small steps, and it decreases fastest (locally, to first order) along `-g`.

Two consequences that matter in practice:

1. The guarantee is **local and first-order**. Curvature, noise and constraints can all
   invalidate the naive picture.
2. The step size `lr` is not a detail. Too large overshoots and diverges; too small is a
   waste of compute and can get stuck in flat regions.

## 1.2 Where the gradient comes from for a linear model

For least squares, `L(w, b) = (1/n) * Σ (w·x_i + b - y_i)^2`, the derivative with respect
to a weight is:

```text
dL/dw_j = (2/n) * Σ (pred_i - y_i) * x_ij
dL/db   = (2/n) * Σ (pred_i - y_i)
```

Written in matrix form, with residuals `r = X w + b - y`:

```text
grad_w = (2/n) * X^T r
```

This is the whole "training loop" of linear regression by gradient descent, and the shape
of it survives unchanged into deep networks: forward to get predictions, residual or
loss-gradient to get the last delta, then backward through the layers.

## 1.3 Batch, stochastic, mini-batch

| Variant | Gradient computed on | Per-step cost | Gradient noise |
|---|---|---|---|
| Batch (full) | all n samples | high | none |
| Stochastic (SGD) | 1 sample | tiny | very high |
| Mini-batch | k samples (64–8192) | medium | tunable |

Mini-batching is the default because it is a trade-off on three axes at once: hardware
efficiency (matrices, not vectors), update frequency per epoch, and gradient noise.

Noise is not automatically bad. With a fixed step size, SGD noise acts like a
regulariser: it prevents the parameters from settling into sharp minima and biases
optimisation toward flatter basins, which typically generalise better. Excessive noise,
though, makes the trajectory wander, so the loss curve looks like static and never
settles.

## 1.4 Preconditioning: momentum and Adam

Plain descent follows the local gradient. In an elongated valley — where one direction is
steep and another is flat — it zig-zags along the steep axis and crawls along the flat
one. Momentum accumulates an exponentially decaying average of past gradients, which
cancels oscillating components and reinforces consistent ones:

```text
v_t  = beta * v_{t-1} + grad
w_{t+1} = w_t - lr * v_t
```

Adam adds a second mechanism: per-coordinate scaling by the running estimate of gradient
magnitude, so flat coordinates get relatively larger steps. AdamW separates weight decay
from the gradient (L2 added into the gradient couples the decay to the adaptive scale,
which is why decoupled decay became the default for transformers).

Rules of thumb that hold up:

- Small data, convex-ish problem: plain GD or L-BFGS-style solvers are often better than Adam.
- Deep nets: AdamW with a warmup and cosine decay is the boring, reliable default.
- Saddle points: SGD-family escapes them because noise breaks symmetry; pure gradient
  descent can stall.

## 1.5 Schedules and diagnostics

A learning rate is a statement about how far you trust local information. Schedules
encode decreasing trust over time:

- **Warmup** — start small while the second-order structure is unknown and activations are
  moving; prevents early divergence.
- **Cosine decay** — smooth reduction toward ~0.
- **Step decay** — divide by 10 at fixed epochs (classic CNN training).
- **One-cycle** — rise then fall, with higher peak LR.

Diagnostics worth keeping as habits:

1. Log the loss every K steps *and* on a fixed validation batch. Divergence in the second
   one is a real signal; a noisy train curve is not.
2. `loss = NaN` → overflow (logits too big, missing clipping, bad loss implementation) or a
   learning rate far too large. Reproduce with a batch of 4 examples first.
3. Loss flat from step 0 → check the gradient norm, not the learning rate. Common causes:
   dead ReLUs, wrong label alignment, forgetting `zero_grad()`, or an unfrozen normalisation
   layer in eval mode.
4. Loss decreasing but validation improving then degrading → you have found the overfitting
   point; use early stopping and note it as the training budget for later runs.

## 1.6 Worked numeric example

Minimise `L(w) = (2w - 6)^2`, starting at `w = 0.5`, `lr = 0.05`.

```text
dL/dw = 2 * (2w - 6) * 2 = 4 * (2w - 6)
at w = 0.5:  4 * (-5) = -20
step:        w <- 0.5 - 0.05 * (-20) = 1.5
next:        dL/dw = 4 * (-3) = -12 -> w = 2.1
```

The minimum is at `w = 3`; with this step size we approach it monotonically. If we set
`lr = 0.6`: `w1 = 0.5 + 12 = 12.5` — we overshoot massively and diverge. The
stability threshold for this quadratic is `lr < 1/(2*second_derivative) = 1/(2*8) = 0.0625`
— note how small that is: the curvature of the objective, not the scale of the loss,
decides the safe step.

## 1.7 Interview-shaped questions to practise

1. Why does gradient descent on a quadratic converge monotonically below a certain step
   size, and what determines that threshold?
2. Why can mini-batch SGD generalise better than full-batch with the same epochs?
3. Why does Adam need decoupled weight decay (AdamW) rather than L2 inside the gradient?
4. Given `L = mean((Xw + b - y)^2)`, derive the gradient in matrix form.
5. Your loss becomes NaN after 200 steps. List four causes and the one-line check for each.
