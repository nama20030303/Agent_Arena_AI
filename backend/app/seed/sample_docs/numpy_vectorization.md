# Vectorisation, Broadcasting and Memory-Bound ML Code

*ML Engineer Academy sample material — self-authored. Chapter 4.*

## 4.1 Why the loop is slow

A Python `for` loop over array elements performs, per element: an iterator step, a type check,
a boxed object allocation, and a dispatch to the numeric operation. NumPy performs one check for
the whole array, then runs a tight C loop over contiguous memory, which the CPU can pipeline and
vectorise with SIMD. The ratio is typically 20–100× for elementwise work, and much larger when the
Python body contains branching.

## 4.2 Broadcasting rules, stated exactly

Two arrays are compatible when, compared from the trailing axis inward, each pair of dimensions is
either equal or one of them is 1. The result's shape is the elementwise maximum. Missing leading
dimensions are treated as 1.

```python
a = np.ones((3, 1, 5))
b = np.ones((     4, 1))
(a + b).shape  # (3, 4, 5)
```

Practical consequences for ML:
- `(N, D) @ (D, 1)` versus `(N, D) * (D,)` — the second broadcasts along rows (correct for
  per-feature transforms); the first would need reshaping. Most silent shape bugs are here.
- Use `X[:, None, :] - C[None, :, :]` for distance matrices — one allocation of `N*K*D`, so it is
  O(N·K·D) memory; for large `K` loop over chunks instead of blowing RAM.
- `keepdims=True` avoids a reshaping step and makes reductions broadcastable again.

## 4.3 Memory bandwidth is the real limit

Modern hardware does ~10–100 GFLOP/s per core but only ~10–20 GB/s per core from DRAM. An operation
touching several `float64` arrays of size n costs more time moving bytes than computing:

```python
z = a * b + c            # reads 3 arrays, writes 1
np.multiply(a, b, out=z) # same math, one temporary saved
z += c                   # fused, no extra allocation
```

Rules that matter in training loops:
1. Prefer in-place ops (`out=`, `+=`) in hot paths — allocation and zero-fill are not free.
2. Contiguity: `X.T @ v` on a Fortran-ordered array is slower; `np.ascontiguousarray` once beats
   N slow calls.
3. Avoid fancy indexing with large index arrays in loops (it copies); sort once and slice.
4. Cast to `float32` where accuracy allows: half the bytes, and SIMD width doubles.
5. Never build a Python list of arrays and `np.stack` it at the end if you can preallocate.

## 4.4 Vectorising the loss and the gradient

Softmax cross-entropy, batch form:

```python
z = logits - logits.max(axis=-1, keepdims=True)
log_p = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
loss = -log_p[np.arange(n), y].mean()
grad = (np.exp(log_p) - one_hot(y)) / n        # (n, C)
```

Two ideas generalise: keep reductions along the axis you intend to remove; and derive the gradient
in matrix form once, then trust the shapes — `X.T @ delta` is the backprop of an affine layer for
any batch size.

## 4.5 When to stop vectorising

- Early-exit / short-circuit logic: a loop that breaks after 3 of 1000 elements beats a full-array
  computation.
- Sparse data: dense broadcasting on mostly-zero matrices wastes memory; use sparse kernels or CSR.
- Very large outputs: chunk over the batch dimension and stream results.
- Debugging: a loop is readable; vectorise after correctness, then diff the two with
  `np.testing.assert_allclose`.

## 4.6 Exercises

1. Implement top-k (return values and indices) without a Python loop, `O(n log k)` or better using
   `np.argpartition`.
2. Vectorise pairwise squared distances between two matrices into an `(N, M)` output using the
   `||a||² + ||b||² - 2a·b` identity, and note the numerical caveat it introduces.
3. Explain why `X - X.mean(axis=0)` is correct for centring but `X - X.mean()` usually is not.
4. Benchmark `for` loop vs vectorised cumulative sum, then vs `np.cumsum`, on 10⁶ floats; explain
   the ratio you observe in terms of allocation and memory traffic.
