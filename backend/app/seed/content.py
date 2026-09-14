"""
Initial seed content (§44): a deliberately small, high-quality starting set.

  * 48 foundational questions (used for the diagnostic and as the bank's seed)
  * 8 math practice tasks
  * 14 coding tasks with real executable tests (many from-scratch)
  * 5 projects with milestones + rubrics
  * 3 exams (junior / middle / senior)

Everything beyond this comes from the learner's library and the generators at runtime.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
#  Questions
# --------------------------------------------------------------------------- #
# (code, topic, skills, level, type, difficulty, stem, options, correct, points, answer, explanation)
def Q(code, topic, skills, level, qtype, diff, stem, options, correct, points, answer, expl, *rest) -> dict[str, Any]:
    """
    Two call shapes:
      conceptual/mcq/open : Q(..., points, model_answer, explanation)
      math               : Q(..., points, model_answer, expected_value, tolerance)
    """
    data: dict[str, Any] = {
        "code": code, "topic": topic, "skills": skills, "level": level, "type": qtype,
        "difficulty": diff, "stem": stem, "options": list(options or []), "correct": correct,
        "points": list(points or []), "answer": answer, "explanation": "",
    }
    if qtype == "math":
        data["expected_value"] = str(expl) if expl is not None else ""
        data["tolerance"] = float(rest[0]) if rest else 0.02
        data["explanation"] = f"Expected value: {data['expected_value']}" + (f" (± {data['tolerance']})" if data["tolerance"] else "")
    else:
        data["explanation"] = expl or ""
    return data


QUESTIONS: list[dict[str, Any]] = [
    # ---------------- level 0: programming foundations
    Q("seed.0.traceback", "foundations.code_literacy", ["code.trace"], 0, "mcq", 1,
      "A Python function raises `IndexError` inside a loop. Where in the traceback is the line that actually caused it?",
      ["The first line of the traceback", "The last line of the traceback", "The middle line", "Tracebacks never show the cause"], 1,
      ["last frame is where the exception was raised", "frames above are the callers"],
      "The last line names the exception; the deepest (last) file/line entry is the raising site. Read bottom-up.",
      "Tracebacks print the call stack outermost-first, so the failing line is at the bottom of the list of frames."),
    Q("seed.0.mutation", "python.core", ["code.types_values"], 0, "mcq", 2,
      "What does this print?\n```python\na = [1, 2, 3]\nb = a[:]\nb.append(4)\nprint(len(a))\n```",
      ["3", "4", "0", "NameError"], 0, ["a[:] copies the list", "mutation of the copy does not affect the original"],
      "3 - the slice creates a shallow copy, so `b.append(4)` does not touch `a`.",
      "Assignment (`b = a`) would alias; slicing copies the container (elements are still shared references)."),
    Q("seed.0.big-o", "foundations.algorithms", ["algo.complexity"], 0, "mcq", 2,
      "For a list of n items already sorted ascending, what is the complexity of binary search, and of `x in list` (worst case)?",
      ["O(log n) and O(n)", "O(n) and O(n)", "O(1) and O(n)", "O(log n) and O(log n)"], 0,
      ["binary search halves the interval", "linear membership scan is O(n)"],
      "Binary search O(log n); `in` on a list O(n). A set would give O(1) average.",
      "This is why hashing (dict/set) is the default trick for membership in feature engineering loops."),
    Q("seed.0.hash-map", "foundations.algorithms", ["algo.collections"], 0, "conceptual", 2,
      "Why is a hash map average O(1) for lookup but O(n) worst case, and what does that mean for caching features by string key?",
      [], None, ["bucket + hash", "collisions degrade to linear", "worst case = adversarial keys"],
      "A hash maps the key to a bucket; if many keys collide the bucket becomes a list and lookup degrades to O(n). For feature caches this means: use stable, well-distributed keys and expect pathological cases when keys share prefixes.",
      "Python dicts are resilient for normal data, but the worst case is real for crafted keys."),
    Q("seed.0.recursion", "foundations.algorithms", ["algo.recursion"], 1, "open", 3,
      "Give the base case and recurrence for computing the sum of a nested list of arbitrary depth. What breaks if the base case is missing?",
      [], None, ["base case: non-list returns value", "recurrence: sum(children)", "stack overflow / infinite recursion"],
      "base: if not isinstance(x, list): return x. recursive: sum(f(child) for child in x). Without a base case, recursion never terminates and Python raises RecursionError.",
      "Same structure as backprop: gradient flows until it reaches a leaf."),
    Q("seed.0.testing", "foundations.testing", ["test.unit"], 0, "mcq", 1,
      "Which test most directly protects a model-serving refactor from silent numeric drift?",
      ["A unit test that the code has docstrings", "A golden-value regression test on a fixed input/output pair", "A lint job", "A load test"], 1,
      ["golden values pin numeric behaviour", "lint/style tests don't check numbers"],
      "Golden-value regression test: assert the refactored code reproduces known outputs within tolerance.",
      "Numeric drift is invisible to type/lint checks; pin the numbers."),

    # ---------------- level 1: python & data
    Q("seed.1.broadcast", "python.numpy", ["np.broadcast"], 1, "mcq", 3,
      "`A` has shape (3, 4) and `v` has shape (4,). What is the shape of `A - v.mean()` and of `A - v`?",
      ["(3, 4) and (3, 4)", "(3, 4) and error", "(4, 3) and (3, 4)", "(3, 1) and (3, 4)"], 0,
      ["trailing-axis alignment", "scalar broadcast keeps shape"],
      "Both are (3, 4): the scalar broadcasts everywhere, and `v` aligns with the last axis of A.",
      "Misalignment errors come from wrong axis, not wrong size - check `A.shape[-1] == v.shape[0]`."),
    Q("seed.1.axis", "python.numpy", ["np.arrays", "np.broadcast"], 1, "mcq", 2,
      "`X` is a (1000, 5) design matrix. Which line centres every feature to mean zero?",
      ["X - X.mean(axis=0)", "X - X.mean(axis=1)", "X - X.mean()", "X / X.std(axis=0)"], 0,
      ["axis=0 reduces over rows", "per-feature means"],
      "X - X.mean(axis=0) subtracts the 5 feature means from every row.",
      "axis=N is the axis that disappears; ML convention is rows=samples, columns=features."),
    Q("seed.1.vectorise", "python.numpy", ["np.broadcast"], 1, "coding", 3,
      "Explain why `for i in range(n): out[i] = f(x[i])` is often 50-100x slower than the vectorised NumPy form, and when vectorising is actually a bad idea.",
      [], None, ["interpretation overhead per element", "SIMD + contiguous memory", "large temporaries / memory bound -> loop better"],
      "Per-element Python has interpreter overhead and no cache locality; NumPy works on contiguous memory with SIMD. Vectorising is a bad idea when it materialises huge intermediates (memory bound) or when the operation short-circuits on the first elements.",
      "The 'vectorise everything' rule has a memory-bandwidth caveat - interviewers like that nuance."),
    Q("seed.1.pandas-dtype", "python.pandas", ["pd.cleaning"], 1, "open", 2,
      "A merge on `df1['id'] == df2['id']` silently produces zero rows. Name the two most likely causes and how you verify each in one line.",
      [], None, ["dtype mismatch int vs object", "trailing whitespace / case", "check dtypes and a sample"],
      "Most often dtypes differ (int64 vs object) or the keys have whitespace/NaN. Verify with `df1.id.dtype, df2.id.dtype` and `df1.id.head().tolist()`.",
      "Always assert dtypes in ETL code - the failure is silent."),
    Q("seed.1.sql-join", "python.sql", ["sql.select"], 1, "mcq", 2,
      "`orders` has 1000 rows, `users` has 200 rows, every order has a valid user_id, and `users.id` is unique. How many rows does `LEFT JOIN` return?",
      ["200", "1000", "1200", "200000"], 1, ["left join preserves left rows", "unique key means no fan-out"],
      "1000 - one row per order because the join key is unique on the right side.",
      "If the right key were not unique you'd get row multiplication - the classic bug in feature backfills."),
    Q("seed.1.sql-window", "python.sql", ["sql.advanced"], 2, "open", 3,
      "Write (pseudocode or real SQL) a query returning, per user, the order amount and the share of that user's total. Which window function do you need and why not a plain GROUP BY?",
      [], None, ["SUM() OVER (PARTITION BY user_id)", "keeps row granularity", "GROUP BY collapses rows"],
      "SELECT user_id, amount, amount * 1.0 / SUM(amount) OVER (PARTITION BY user_id) AS share FROM orders; A GROUP BY would collapse to one row per user, losing per-order detail.",
      "Window functions aggregate without changing the row grain - the core skill for feature SQL."),
    Q("seed.1.stats-variance", "python.statistics", ["stat.describe"], 1, "math", 2,
      "Data: [2, 4, 4, 4, 5, 5, 7, 9]. Compute the population variance (divide by n).",
      [], None, ["mean = 5", "squared deviations", "divide by n=8"], "4.0", "4", 0.01),
    Q("seed.1.stats-ci", "python.statistics", ["stat.inference"], 2, "mcq", 3,
      "You double the sample size of an A/B test. Approximately what happens to the width of a 95% CI on the mean difference?",
      ["Halves", "Divides by sqrt(2)", "Is unchanged", "Doubles"], 1,
      ["SE scales as 1/sqrt(n)", "width proportional to SE"],
      "Width shrinks by a factor sqrt(2) ~ 0.707, not by 2 - the classic diminishing-returns law of experiments.",
      "To halve the width you need 4x the data."),
    Q("seed.1.stats-interpret", "python.statistics", ["stat.inference"], 2, "open", 3,
      "A test gives p = 0.03 for a +0.4% conversion lift on 2M users. Is it worth shipping? Explain the reasoning you would show a VP.",
      [], None, ["statistical vs practical significance", "effect size CI", "cost/risk of change", "guardrail metrics"],
      "Significance is not importance: report the CI on the lift (e.g. 0.4% [0.05%, 0.75%]), compare to the cost and risk, check guardrail metrics, and consider novelty/Simpson effects by segment.",
      "With huge n almost anything is 'significant'; decide on effect size and cost."),
    Q("seed.1.comprehension", "python.core", ["py.idioms"], 1, "coding", 2,
      "Write a one-liner building `{word: length}` for a list `words`, keeping only words longer than 3 characters.",
      [], None, ["dict comprehension", "condition filter"],
      "{w: len(w) for w in words if len(w) > 3}",
      "Comprehensions with a filter clause are the idiomatic, faster-than-loop form."),

    # ---------------- level 2: mathematics
    Q("seed.2.dot", "math.linear_algebra", ["linalg.vectors"], 2, "math", 1,
      "Compute the dot product of a=[1, 2, 3] and b=[4, 0, 2].",
      [], None, ["multiply elementwise", "sum"], "10", "10", 0.01),
    Q("seed.2.norm", "math.linear_algebra", ["linalg.vectors"], 2, "math", 2,
      "Compute the L2 norm of the vector [3, -4, 12] (round to 2 decimals).",
      [], None, ["square each", "sum", "sqrt"], "13.0", "13", 0.02),
    Q("seed.2.matmul-shape", "math.linear_algebra", ["linalg.matrices"], 2, "mcq", 2,
      "A is (m, n) and B is (n, p). What is the shape of A @ B, and what does entry (i, j) mean?",
      ["(m, p); dot of row i of A with column j of B", "(n, n); elementwise product", "(p, m); dot of column i with row j", "(m, n, p); outer product"], 0,
      ["inner dimensions must match", "row-column dot"],
      "(m, p), where (i,j) is the dot product of row i of A and column j of B.",
      "Linear layer = matmul; shapes are the first thing to debug in any net."),
    Q("seed.2.eigen", "math.linear_algebra", ["linalg.decomposition"], 3, "open", 4,
      "Why are eigenvectors of the covariance matrix the natural axes for PCA? Mention the optimisation being solved.",
      [], None, ["variance = v^T S v", "maximised by top eigenvector", "orthogonality from symmetry", "Rayleigh quotient"],
      "PCA maximises projected variance v^T Σ v subject to ||v||=1; the symmetric eigen-decomposition makes the top eigenvector the argmax, and eigenvectors of a symmetric matrix are orthogonal, so components are decorrelated.",
      "SVD of the centred data is the numerically preferred way to compute it."),
    Q("seed.2.derivative", "math.calculus", ["calc.derivatives"], 2, "math", 2,
      "f(x) = x^3 - 4x. Compute f'(2).",
      [], None, ["power rule", "3x^2 - 4"], "8", "8", 0.01),
    Q("seed.2.partial", "math.calculus", ["calc.partial_gradient"], 2, "math", 3,
      "L(x, y) = x^2 y + sin(y). Compute dL/dx at (1, 2) (give a numeric value to 3 decimals).",
      [], None, ["treat y constant", "2xy"], "4.0", "4", 0.01),
    Q("seed.2.chain", "math.calculus", ["calc.partial_gradient", "dl.backprop"], 3, "open", 3,
      "State the chain rule for L = f(g(x)) and explain how it becomes backpropagation in a 2-layer network.",
      [], None, ["dL/dx = dL/df * df/dg * dg/dx", "gradients flow backwards", "reuse of intermediate activations"],
      "The gradient of a composition is the product of the local gradients; a network is a composition of affine maps and non-linearities, so backprop is a reverse-mode traversal multiplying local Jacobians and caching activation gradients.",
      "Reverse mode costs one forward + one backward pass regardless of the number of parameters."),
    Q("seed.2.gradient-meaning", "math.calculus", ["calc.partial_gradient", "opt.gradient_descent"], 2, "conceptual", 2,
      "Geometrically, which direction does the gradient point, and why does gradient descent move in the opposite one?",
      [], None, ["steepest ascent", "negative direction decreases fastest locally", "first-order approximation"],
      "The gradient points in the direction of steepest local increase, so the negative gradient is the locally steepest descent direction under the linear approximation L(w - t g) ~ L(w) - t||g||^2.",
      "Only local and first-order: step size and curvature decide whether it actually helps."),
    Q("seed.2.prob-basics", "math.probability", ["prob.basics"], 2, "math", 2,
      "Two fair dice are rolled. What is the probability that the sum is 8? Give a decimal rounded to 4 places.",
      [], None, ["36 outcomes", "5 favourable", "5/36"], "0.1389", "0.1389", 0.002),
    Q("seed.2.bayes", "math.probability", ["prob.basics"], 3, "math", 4,
      "A disease affects 1% of people. A test has 99% sensitivity and 5% false positive rate. Compute the posterior probability that a positive patient is sick (in %, 2 decimals).",
      [], None, ["P(+|sick)=0.99", "P(+|healthy)=0.05", "divide by total probability", "base rate matters"],
      "16.67", "16.67", 0.5),
    Q("seed.2.expectation", "math.probability", ["prob.random_vars"], 2, "mcq", 2,
      "If X has mean 3 and variance 4, what are E[2X+1] and Var[2X+1]?",
      ["7 and 16", "7 and 8", "6 and 16", "7 and 4"], 0,
      ["E[aX+b]=aE[X]+b", "Var[aX+b]=a^2 Var[X]"],
      "E = 2*3+1 = 7; Var = 2^2 * 4 = 16.",
      "Variance scales quadratically - the reason feature scaling changes regularisation strength."),
    Q("seed.2.entropy", "math.info_theory", ["info.entropy"], 3, "math", 3,
      "Compute the Shannon entropy (bits) of a coin with p(head)=0.8, to 3 decimals. Use H = -[p log2 p + (1-p) log2 (1-p)].",
      [], None, ["log2 of 0.8 and 0.2", "weighted sum", "negative sign"], "0.722", "0.722", 0.005),
    Q("seed.2.xent", "math.info_theory", ["info.entropy", "ml.logistic_regression"], 3, "open", 3,
      "Why is cross-entropy the natural loss for softmax classification rather than MSE?",
      [], None, ["log-likelihood of the true class", "gradient = p - y, no vanishing", "MSE + softmax flattens gradients"],
      "Cross-entropy is the negative log-likelihood; its gradient w.r.t. logits is exactly (p - y), which stays well-scaled when the model is confidently wrong. MSE through a softmax saturates and produces tiny gradients.",
      "Also connects to KL divergence up to a constant."),
    Q("seed.2.numstable", "math.numerical", ["num.stability"], 3, "open", 3,
      "Why does computing softmax as exp(z_i)/sum(exp(z_j)) overflow, and what is the fix?",
      [], None, ["large logits overflow exp", "subtract max logit", "invariance to shift"],
      "Subtract z_max before exponentiating; softmax is shift-invariant so the value is unchanged while the exponentials stay <= 1.",
      "Same trick (log-sum-exp) applies to log-softmax and attention."),

    # ---------------- level 3: classical ML
    Q("seed.3.leakage", "ml.framing", ["ml.data_splits"], 3, "mcq", 3,
      "Which of these is data leakage?",
      ["Fitting a scaler on the full dataset before splitting", "Stratified k-fold CV", "Early stopping on validation loss", "Hyperparameter grid search inside CV"], 0,
      ["transform must be fit on train only", "test/val must stay untouched"],
      "Fitting the scaler (or imputer, encoder) on all data before splitting leaks distribution information into the evaluation.",
      "The fix is a Pipeline: the transform is fit inside each fold."),
    Q("seed.3.bias-variance", "ml.bias_variance", ["ml.bias_variance"], 3, "open", 3,
      "Training error is 1% and validation error is 28%. Diagnose and give three fixes ordered by how cheap they are to try.",
      [], None, ["high variance / overfitting", "regularisation", "more data", "simpler model / dropout or weight decay"],
      "Classic overfitting. Cheapest first: (1) stronger regularisation (ridge/lasso alpha, dropout, weight decay), (2) reduce model complexity / prune features, (3) add data or augmentation. Also verify the split is not leakage-driven.",
      "If both errors were high it would be bias/underfitting - opposite prescription."),
    Q("seed.3.linreg-solve", "ml.linear_models", ["ml.linear_regression", "linalg.least_squares"], 3, "math", 3,
      "Points: (0,1), (1,3), (2,5). Fit y = a + bx by least squares. Give a and b.",
      [], None, ["normal equations or closed form for 2 params", "b = covariance/variance", "a = ybar - b xbar"],
      "a=1, b=2", "a=1, b=2", 0.01),
    Q("seed.3.logistic", "ml.linear_models", ["ml.logistic_regression"], 3, "mcq", 2,
      "Logistic regression models P(y=1|x) with sigmoid(w·x+b). What does a coefficient of 0.7 mean?",
      ["The probability increases by 0.7", "The log-odds increase by 0.7 per unit of that feature", "The class is 70% likely", "The feature explains 70% of variance"], 1,
      ["coefficients act on log-odds", "odds ratio = exp(w)"],
      "It is a log-odds shift: the odds multiply by exp(0.7) ~ 2.01 per unit increase, holding others fixed.",
      "Never read a logit coefficient as a probability change."),
    Q("seed.3.gd-lr", "ml.gradient_descent", ["ml.gradient_descent_impl", "opt.gradient_descent"], 3, "mcq", 3,
      "You increase the learning rate far beyond a stable value. What do you usually see in the loss curve?",
      ["Smooth monotonic decrease", "Loss oscillates then diverges / becomes NaN", "Loss immediately becomes constant", "Nothing changes"], 1,
      ["overshooting the minimum", "divergence/NaN from exploding gradients"],
      "Too-large steps overshoot, oscillate with growing amplitude and typically blow up to NaN.",
      "Gradient clipping + warmup + LR scheduling are the standard mitigations."),
    Q("seed.3.minibatch", "ml.gradient_descent", ["ml.optimizers"], 4, "open", 4,
      "Why does mini-batch SGD often generalise better than full-batch gradient descent with the same number of epochs?",
      [], None, ["gradient noise acts as regulariser", "flatter minima", "more updates per epoch"],
      "Mini-batches inject label/gradient noise, which acts like regularisation and biases optimisation toward flatter minima; they also give strictly more parameter updates for a fixed epoch budget.",
      "Batch size interacts with LR (linear scaling rule) - mention that too for a strong answer."),
    Q("seed.3.metrics", "ml.model_evaluation", ["ml.metrics"], 3, "math", 3,
      "Confusion matrix: TP=90, FP=10, FN=30, TN=70. Compute precision and recall (decimals, 2 places).",
      [], None, ["precision = TP/(TP+FP)", "recall = TP/(TP+FN)"],
      "precision=0.90, recall=0.75", "0.90, 0.75", 0.01),
    Q("seed.3.roc-auc", "ml.model_evaluation", ["ml.metrics"], 3, "mcq", 4,
      "Your dataset has 1% positives. Which metric tells you the most about ranking quality on the positive class?",
      ["Accuracy", "ROC AUC", "Precision@K / PR AUC", "Log-loss"], 2,
      ["ROC looks optimistic under heavy imbalance", "PR curve focuses on positives"],
      "PR AUC / Precision@K: with 99% negatives, ROC AUC and accuracy both look great while the model may still be useless at the operating point.",
      "Report the operating point (threshold + volume + precision) you will actually ship."),
    Q("seed.3.cv", "ml.model_evaluation", ["ml.data_splits", "adv.nested_cv"], 3, "open", 3,
      "You tuned hyperparameters with 5-fold CV and reported the best fold's score as 'the model score'. What is wrong and what is the correct protocol?",
      [], None, ["selection bias / CV overfitting", "nested CV or holdout test set", "report mean +- std of outer folds"],
      "Choosing the best config on the same folds biases the estimate upward. Use nested CV (or an untouched holdout) and report the outer-fold mean and spread.",
      "Outer loop estimates performance, inner loop chooses hyperparameters."),
    Q("seed.3.trees", "ml.tree_models", ["ml.trees"], 3, "mcq", 2,
      "A decision tree chooses a split by maximising information gain. Information gain is:",
      ["Parent entropy minus the weighted child entropies", "The child entropy minus the parent entropy", "Gini times depth", "The number of leaves divided by samples"], 0,
      ["reduction in impurity", "weighted by child sizes"],
      "IG = H(parent) - sum_k (n_k/n) H(child_k).",
      "Gini is the same idea with a different impurity function; both push toward pure leaves."),
    Q("seed.3.bagging-boosting", "ml.tree_models", ["ml.ensembles", "ml.boosting"], 4, "open", 4,
      "Contrast bagging and gradient boosting: what each reduces, and how they use model capacity.",
      [], None, ["bagging reduces variance via decorrelated averaging", "boosting reduces bias via residual fitting", "parallel vs sequential", "shrinkage learning rate"],
      "Bagging trains many deep/uncorrelated models in parallel and averages them - variance reduction. Boosting fits weak learners sequentially to residuals/negative gradients - bias reduction, controlled by shrinkage and early stopping.",
      "That is why forests are robust to more trees while boosting degrades if you over-iterate."),
    Q("seed.3.kmeans", "ml.clustering", ["ml.clustering_kmeans"], 3, "open", 3,
      "k-means objective, and two failure modes with one mitigation each.",
      [], None, ["minimise within-cluster sum of squares", "non-convex clusters", "bad init / k choice", "kmeans++ and silhouette / GMM"],
      "Minimise sum of squared distances to assigned centroids. Fails on non-convex or varying-density clusters (use DBSCAN/spectral) and on poor initialisation (use kmeans++ + multiple restarts, and pick k by silhouette/elbow).",
      "Lloyd's algorithm monotonically decreases the objective but to a local optimum."),
    Q("seed.3.regularization", "ml.bias_variance", ["ml.regularization"], 4, "mcq", 3,
      "Why does L1 (lasso) produce exact zeros while L2 (ridge) does not?",
      ["L1 penalty is non-differentiable at zero with a sharper contour", "L2 is convex and L1 is not", "L1 uses a smaller learning rate", "Because of feature scaling"], 0,
      ["diamond-shaped constraint / subgradient at 0", "soft-thresholding"],
      "The L1 ball has corners on the axes and the subgradient at zero is discontinuous, so solutions get pinned exactly at 0 - sparsity.",
      "L2 shrinks coefficients smoothly instead."),
    Q("seed.3.imbalanced", "adv.imbalanced", ["adv.imbalance", "adv.cost_sensitive"], 4, "open", 4,
      "Fraud detection: 0.2% positives, you need >95% recall at a precision >= 20% operating point. Describe your approach end to end.",
      [], None, ["choose metric + threshold by cost", "resample/weight", "calibration", "eval on PR + volume constraints", "human review capacity"],
      "Optimise PR AUC at the target recall, use class weights or SMOTE-style resampling cautiously, calibrate probabilities (Platt/isotonic) since thresholding depends on them, search the threshold under the precision floor, and validate against the actual review-team capacity; monitor score drift online.",
      "Never report accuracy on this problem."),

    # ---------------- level 4-5: advanced ML / DL
    Q("seed.4.optuna", "adv.tuning", ["adv.tuning_skills"], 4, "mcq", 3,
      "Bayesian hyperparameter search beats random search mainly when:",
      ["The search space is huge and cheap to sample", "Evaluations are expensive and the response surface is smooth", "The model is linear", "You have unlimited compute"], 1,
      ["expensive evaluations", "surrogate model exploits structure"],
      "With costly evaluations a surrogate model (GP/TPE) exploits previous results; random search is nearly as good when evaluations are cheap.",
      "Always log all trials - that is what makes the next round smarter."),
    Q("seed.5.neuron", "dl.foundations", ["dl.neuron"], 5, "mcq", 2,
      "What happens if you stack 5 linear layers with no non-linearity between them?",
      ["The network becomes exponentially deeper and better", "It collapses to a single linear transformation", "It becomes a CNN", "Gradients vanish to zero always"], 1,
      ["composition of linear maps is linear", "no added expressive power"],
      "The composition of affine maps is one affine map - depth without non-linearity buys nothing.",
      "Non-linearity is what makes depth useful."),
    Q("seed.5.vanishing", "dl.backprop", ["dl.vanishing"], 5, "open", 4,
      "Explain vanishing gradients in a deep sigmoid network and list four independent fixes.",
      [], None, ["product of sub-unit derivatives", "sigmoid saturation", "residual connections", "ReLU/gelu", "normalisation layers", "proper init + gradient clipping"],
      "Each layer multiplies the gradient by its local derivative; sigmoids have |f'| <= 0.25 so the product decays exponentially. Fixes: ReLU-family activations, residual connections, normalisation (Batch/LayerNorm), careful initialisation (He/Xavier), gradient clipping + warmup, and pretraining/curriculum.",
      "Exploding gradients are the symmetric failure; clipping is the shared band-aid, norm + init are the cure."),
    Q("seed.5.init", "dl.foundations", ["dl.init_norm"], 5, "mcq", 3,
      "He initialisation (var = 2/fan_in) is derived so that:",
      ["Weights sum to 1", "Forward activation variance is preserved across layers for ReLU", "Loss starts at 0.69", "Gradients are exactly equal for all layers"], 1,
      ["variance propagation", "ReLU halves the variance -> factor 2"],
      "It preserves signal variance through ReLU layers (the 2 compensates ReLU dropping half the activations).",
      "Xavier/Glorot is the tanh version without the factor 2."),
    Q("seed.5.bn", "dl.training", ["dl.regularization", "dl.init_norm"], 5, "open", 4,
      "BatchNorm: what it normalises, why it speeds training, and the one gotcha at inference time.",
      [], None, ["per-feature batch statistics", "reduces internal covariate shift / smoother loss", "running mean-var for eval", "small batches hurt", "needs train/eval mode"],
      "BN standardises each channel's activations using batch statistics, giving a smoother loss landscape and larger tolerable learning rates; at inference it must use the running averages (model.eval()!), and with tiny batches its statistics become noisy.",
      "The train/eval mismatch is the most common BN bug in production."),
    Q("seed.5.dropout", "dl.training", ["dl.regularization"], 4, "mcq", 3,
      "Standard dropout at inference time applies:",
      ["The same random mask", "No mask; weights are already scaled during training (inverted dropout)", "x2 scaling", "BatchNorm instead"], 1,
      ["inverted dropout scales during training", "expectation-preserving"],
      "Inverted dropout scales activations by 1/(1-p) during training so inference is the plain forward pass.",
      "Forgetting to switch to eval mode changes results on BN and dropout models."),
    Q("seed.5.losscurve", "dl.training", ["dl.schedules", "ml.bias_variance"], 5, "open", 4,
      "Training loss decreases smoothly, validation loss decreases then rises steadily. Diagnose and give a treatment plan.",
      [], None, ["overfitting after some epoch", "early stopping at min", "regularisation/augmentation", "reduce capacity or increase data", "check for duplicate leakage"],
      "Textbook overfitting: stop at the validation minimum (early stopping with patience), increase augmentation/regularisation, reduce capacity or add data, and verify no train/eval leakage inflates the curve.",
      "Also check the LR schedule: a too-high LR can look like this too."),
    Q("seed.5.autograd", "dl.backprop", ["dl.backprop_impl"], 5, "coding", 4,
      "Write the backward pass for y = relu(W @ x + b) given dL/dy. Which quantities do you need from the forward pass, and why?",
      [], None, ["cached pre-activation or relu mask", "dx = W^T dy * mask", "dW = outer(dy*mask, x)"],
      "d_pre = dL/dy * (pre > 0); dL/dx = W^T d_pre; dL/dW = outer(d_pre, x); dL/db = d_pre.sum(). You must cache the pre-activation (or the mask) from the forward pass because the derivative depends on it.",
      "That is exactly what an autograd tape stores per node."),
    Q("seed.5.cnn-params", "dl.cnn", ["dl.cnn"], 5, "math", 3,
      "A conv layer: 3 input channels, 16 filters of 3x3, 'same' padding, 224x224 input. How many trainable weights (including biases)?",
      [], None, ["k*k*Cin*Cout", "plus Cout biases"], "448", "16*3*3*3 + 16 = 432 + 16 = 448", 0.5),
    Q("seed.5.pytorch-device", "dl.pytorch", ["dl.pytorch_core"], 5, "open", 3,
      "A RuntimeError says 'Expected all tensors to be on the same device'. Give three plausible causes.",
      [], None, ["buffer/const created without device", "model not .to(device)", "batch tensor left on CPU by custom collate", "checkpoint loaded on cpu then used with cuda"],
      "Usually: (1) module-level `torch.zeros(1)` buffers not moved, (2) model built before .to(device), (3) a custom DataLoader/collate or manually created target staying on CPU, or state_dict loaded with map_location cpu and never moved.",
      "Standardise one `device` variable per process and pass it everywhere."),

    # ---------------- level 6: NLP / LLM
    Q("seed.6.bpe", "nlp.tokenization", ["nlp.tokenizers"], 6, "open", 3,
      "Why do subword tokenisers (BPE/SentencePiece) make out-of-vocabulary words impossible but create new problems for counting/classification tasks?",
      [], None, ["merges from corpus frequency", "no UNK", "word-piece artefacts", "inflated sequence lengths", "subword-level labels mismatch"],
      "Merging from a corpus builds a fixed vocabulary of subwords so any string is representable. Costs: sequence length inflation, split artefacts (numbers/dates broken), and the fact that one 'word' may be several tokens - which breaks token-classification alignment and biases length-based features.",
      "Always check the tokenizer's handling of digits and unicode in your domain."),
    Q("seed.6.attention", "nlp.transformers", ["nlp.attention"], 6, "math", 4,
      "Attention is softmax(QK^T / sqrt(d_k)) V. Why divide by sqrt(d_k)?",
      [], None, ["dot products grow with dim", "keep logits in sane range", "avoid saturated softmax"],
      "If q and k have unit variance per component, q.k has variance ~d_k; dividing by sqrt(d_k) keeps the variance at ~1 so softmax does not saturate and gradients stay healthy.",
      "Same reasoning as scaled initialisation in MLPs."),
    Q("seed.6.kvcache", "nlp.transformers", ["nlp.kv_cache"], 7, "open", 4,
      "Explain the KV cache: what is cached, why it saves compute, and what it costs in memory.",
      [], None, ["K,V of past tokens", "one token decode per step", "memory grows with batch*seq*layers*heads*dim*2", "eviction/paging needed"],
      "During autoregressive decoding the K and V of previous tokens never change, so they are cached and only the new token's QKV is computed; the cost is a per-sequence buffer of 2 * layers * heads * head_dim * seq_len floats, which is why long-context serving is memory-bound (and why PagedAttention/KV quant exist).",
      "Prefill vs decode have completely different bottlenecks."),
    Q("seed.6.rag-eval", "nlp.rag", ["rag.grounding", "llm.evals"], 7, "open", 4,
      "How would you evaluate a RAG system so you can tell whether a bad answer is a retrieval failure or a generation failure?",
      [], None, ["separate retrieval metrics (recall@k, MRR)", "answer faithfulness vs context", "answer relevancy", "ablations with gold context", "judge calibration"],
      "Two-layer eval: retrieval (recall@k / MRR / nDCG on a labelled query-doc set) and generation (faithfulness/groundedness against retrieved context + answer relevancy). Then run an oracle-context ablation: feed the gold passages and re-measure - if the score jumps, it is retrieval; if it does not, it is generation/prompting.",
      "Also measure refusal rate: a good RAG says 'not in the sources'."),
    Q("seed.6.chunking", "nlp.rag", ["rag.chunking"], 6, "open", 3,
      "Choose chunk size/overlap for a technical PDF with formulas and code blocks, and justify it.",
      [], None, ["structure-aware boundaries", "keep code/formula blocks atomic", "300-800 tokens typical", "overlap small", "metadata per chunk"],
      "Split on headings/paragraphs rather than raw characters, keep code fences and equations atomic, target ~300-600 tokens so the retriever can score a self-contained unit, use a small overlap (10-15%) only across paragraph breaks, and attach heading/page metadata so answers can be cited.",
      "Oversized chunks destroy precision; undersized ones destroy context."),
    Q("seed.6.hallucination", "nlp.rag", ["rag.grounding", "mle.security_ml"], 6, "mcq", 3,
      "Which measure most directly reduces hallucinated citations in a RAG answer?",
      ["Lowering temperature to 0", "Requiring spans/IDs from the retrieved set to be echoed and validated post-hoc", "Increasing max tokens", "Adding more system-prompt instructions"], 1,
      ["programmatic validation of references", "generation alone is not a guarantee"],
      "Constrain the model to emit source IDs and validate them against the retrieved set (reject/repair when unknown), rather than trusting prose instructions.",
      "Temperature and prompt text help; structural validation is what actually catches it."),
    Q("seed.6.lora", "nlp.pretraining", ["nlp.peft"], 6, "mcq", 3,
      "LoRA fine-tuning freezes the base weights and trains:",
      ["Full-rank copies of every layer", "Low-rank update matrices A·B on selected projections", "Only the embedding layer", "The layer norms only"], 1,
      ["rank decomposition", "few trainable params", "mergeable at inference"],
      "LoRA trains low-rank delta W = B·A on (usually) q/v projections, keeping the base frozen; the update can be merged so inference cost is unchanged.",
      "Memory savings come mostly from not storing optimiser state for frozen weights."),

    # ---------------- level 7-9: engineering / MLOps / design
    Q("seed.7.serving", "mle.serving", ["mle.api", "mle.serving_perf"], 7, "open", 3,
      "Your FastAPI inference endpoint has p50 = 8ms but p99 = 400ms on a 4-core box. Give four plausible causes and the measurement you'd make first.",
      [], None, ["GIL / CPU-bound in handler", "no batching under load", "cold model / lazy import", "GC pauses", "thread pool exhaustion", "measure with concurrency + py-spy"],
      "Measure p99 vs concurrency curve and profile with py-spy under load first. Usual suspects: blocking CPU work in an async handler (starves the event loop), no dynamic batching so tail requests queue, model warmup not done (lazy compile), GC/allocator pauses, and thread-pool size limits under I/O.",
      "Fix order: warmup -> move to threadpool/processes -> micro-batching -> capacity/autoscaling."),
    Q("seed.7.contracts", "mle.data_contracts", ["mle.contracts"], 7, "mcq", 3,
      "A model's score drops on Monday with no deploy. First thing to check?",
      ["Re-run training", "Feature distribution and schema of the incoming data vs training window", "Increase learning rate", "Add more GPUs"], 1,
      ["input drift/schema change is the commonest cause", "before retraining"],
      "Compare online feature distributions and schema/null rates against the training snapshot - drift or a broken upstream field is the most frequent silent regression.",
      "Only after the data is clean does retraining make sense."),
    Q("seed.7.repro", "mle.package_deploy", ["mle.deploy", "tools.env"], 7, "open", 3,
      "List what you pin to make a training run reproducible enough to re-issue a shipped model months later.",
      [], None, ["code commit + container image digest", "data version + snapshot", "seeds and deterministic flags", "hyperparameters + env", "hardware/precision notes"],
      "Pin: git commit, image digest, dependency lock, dataset version (DVC/manifest hash), seeds + determinism flags, hyperparameters/config, feature-store version, and note hardware/precision since TF32 vs FP32 changes results.",
      "Reproducibility is what makes an audit or a rollback possible."),
    Q("seed.8.monitoring", "mlops.monitoring", ["mlops.monitoring_ops"], 8, "open", 4,
      "Design the minimum useful monitoring set for a production ranking model, including the alert you would NOT set.",
      [], None, ["input/feature health", "prediction distribution", "label-delayed quality", "latency/error budget", "no alert on raw accuracy for tiny traffic"],
      "Track feature null rates and drift (PSI/KS), score distribution and top-k stability, latency p50/p99 and error rate, plus delayed label-based quality with confidence intervals. Do NOT alert on accuracy computed from a handful of labels: on low-traffic slices it is pure noise and generates pager fatigue.",
      "Alert on sustained, statistically meaningful deviations and on missing data, not on point estimates."),
    Q("seed.8.retraining", "mlops.automation", ["mlops.retraining", "mlops.release"], 8, "mcq", 4,
      "Automated retraining should promote a new model to production when:",
      ["Its training loss is lower than the incumbent's", "It beats the incumbent on a frozen evaluation window and passes shadow/canary checks", "It is newer", "Its GPU usage is lower"], 1,
      ["frozen evaluation set", "online validation before promotion", "rollback path"],
      "Promotion needs an offline win on a fixed, leakage-free evaluation window AND online validation (shadow -> canary) with an automatic rollback trigger.",
      "Never promote on training loss or on recency."),
    Q("seed.9.system-design", "design.api_systems", ["design.requirements", "design.tradeoffs"], 9, "architecture", 5,
      "Design a near-real-time product recommendation service: 30M users, 50k RPS, p99 < 80ms, model refreshed hourly, cost is a hard constraint. Describe retrieval/ranking split, freshness, storage, fallback and capacity math.",
      [], None, ["two-stage candidate generation + ranking", "precompute + cache", "point-in-time online features", "degradation ladder", "capacity/QPS math", "cost per request"],
      "Strong answer structure: (1) precompute user embeddings and ANN index shards; hourly job rebuilds candidates, per-request work is a vector search + a light ranker; (2) cache top-k per user with TTL and stampede protection; (3) online features from a low-latency store (sub-ms reads, nearline feature pipeline for the rest); (4) explicit fallback: cached-then-popular; (5) capacity math: 50k RPS * per-request CPU/GPU time -> instances, headroom 30-40%, autoscale on p99; (6) cost levers: quantised vectors, batched GPU inference, spot for offline; (7) monitoring + rollback; (8) freshness/quality trade-off justified with metrics.",
      "Interviewers score the trade-off reasoning, not the number of components."),
    Q("seed.9.cost", "design.cost", ["design.cost_model", "design.optimization_cost"], 9, "open", 4,
      "An LLM-based internal tool costs 6x more than the budget. Give the optimisation plan ordered by expected savings/risk.",
      [], None, ["caching/prompt dedup", "shorter prompts + retrieved context pruning", "smaller model + router", "quantisation/batching", "offline/batch API for non-interactive", "measure tokens per task"],
      "1) Cache and dedupe (identical/semantic) - often the largest single win, near-zero risk. 2) Cut the prompt: fewer retrieved chunks, structured instructions, cap output tokens. 3) Route: cheap model by default, escalate on uncertainty/confidence; validate quality per route. 4) Serving: quantisation + continuous batching if self-hosted. 5) Move non-interactive work to batch endpoints. Measure cost per completed task, not per call.",
      "Always re-run the eval harness after each lever; savings that destroy quality are not savings."),
    Q("seed.9.incident", "senior.incidents", ["senior.incident_leadership"], 10, "open", 5,
      "At 03:00 your fraud model's precision collapses from 0.9 to 0.3 and it is auto-blocking payments. What do you do in the next 30 minutes, and what do you change afterwards?",
      [], None, ["stop the bleeding / rollback or safe mode", "communicate, define blast radius", "verify data upstream cause", "do not retrain mid-incident", "postmortem + guardrails on automated actions"],
      "First mitigate: switch to the safe policy (manual review threshold / previous model version), because the cost of false blocks is immediate; freeze the state for forensics, notify stakeholders with blast radius, check upstream data quality/schema, and explicitly avoid retraining during the incident. Afterwards: postmortem, add a guardrail so a metric deviation auto-degrades the policy, plus replay-based release gates.",
      "Senior signal: reversible-by-design automation and clear incident discipline."),
    Q("seed.10.leadership", "senior.ownership", ["senior.ownership", "senior.evaluation_culture"], 10, "open", 5,
      "You must convince two product teams to fund an evaluation platform instead of new models. What is your argument and what do you measure to prove it?",
      [], None, ["cost of bad decisions", "regression risk", "shared eval + golden sets", "cycle time", "explicit non-goals", "adoption metric"],
      "Frame it as decision throughput: without a shared eval harness every team rebuilds the same 3 scripts, offline wins do not transfer, and rollbacks are expensive. Measure: time from idea to trustworthy verdict, % of releases with a documented eval gate, number of caught regressions, and reuse across teams. Offer a thin interface (dataset registry + scorer API), not a platform monolith, and name what you will NOT build.",
      "Senior work is buying leverage for other people, not owning more code."),
    Q("seed.3.sklearn-pipeline", "ml.preparation", ["ml.pipelines"], 3, "mcq", 2,
      "In `Pipeline([('prep', ct), ('model', clf)])`, what does `pipeline.fit(X_train, y_train)` do?",
      ["Fits transformers on all data to save time", "Fits each step on the output of the previous step using only the passed data", "Only fits the final estimator", "Serialises the model"], 1,
      ["fit_transform chaining", "no leakage by construction"],
      "Each preprocessing step is fit on X_train and its transform is passed to the next step, so CV folds can't leak.",
      "That's the mechanical reason pipelines beat hand-written preprocessing loops."),
    Q("seed.4.calibration", "adv.fairness", ["adv.fairness_metrics", "ml.calibration"], 4, "open", 4,
      "Your model has near-identical AUC for two groups but very different false-positive rates. Is that a problem, and which fairness framings conflict here?",
      [], None, ["equalised odds vs calibration conflict", "base rate differences", "choose the constraint explicitly", "document the trade-off"],
      "Yes: equal AUC does not imply equal error rates. Calibration and equalised-odds constraints are mathematically incompatible when base rates differ, so you must choose which constraint the product commits to, document it, and monitor the chosen metric per group with confidence intervals.",
      "Naming the impossibility result is the senior answer."),
    Q("seed.5.batchnorm-vs-layernorm", "dl.training", ["dl.init_norm"], 5, "mcq", 3,
      "Which statement about LayerNorm vs BatchNorm is correct?",
      ["LayerNorm normalises across the batch, BatchNorm across features", "LayerNorm normalises per-sample over features, so it behaves identically in train and eval", "BatchNorm has no train/eval difference", "LayerNorm requires larger batches"], 1,
      ["per-sample statistics", "no running stats needed"],
      "LayerNorm computes statistics per sample over the feature axis, so it is batch-size independent and has no train/eval mismatch - which is why it dominates in transformers.",
      "BatchNorm's batch coupling is also why it is awkward in small-batch or streaming settings."),
]


# --------------------------------------------------------------------------- #
#  Math / practice tasks (§19 ladder per topic)
# --------------------------------------------------------------------------- #
MATH_TASKS: list[dict[str, Any]] = [
    {
        "topic": "ml.gradient_descent", "skills": ["opt.gradient_descent", "ml.gradient_descent_impl"], "level": "beginner",
        "kind": "math", "difficulty": 2, "est_minutes": 8,
        "title": "One manual gradient-descent step",
        "statement": "L(w) = (w*2 - 6)^2 with w = 0.5. Compute dL/dw and one update with learning rate 0.05. Give w_new.",
        "expected_value": "1.5", "tolerance": 0.01,
        "steps_required": ["expand or chain-rule the derivative", "dL/dw = 2*(2w-6)*2", "w <- w - lr*g"],
        "solution": "dL/dw = 4*(2w-6) = 4*(-5) = -20. w_new = 0.5 - 0.05*(-20) = 1.5.",
        "explanation": "Check the sign: g is negative so w increases toward the minimum at w=3.",
    },
    {
        "topic": "ml.linear_models", "skills": ["ml.linear_regression"], "level": "intermediate",
        "kind": "math", "difficulty": 3, "est_minutes": 10,
        "title": "Closed-form linear regression on 2 points",
        "statement": "With X = [[1],[2]], y = [3, 5] and an intercept, compute the least-squares slope b via the covariance/variance formula (x means included). Give b only.",
        "expected_value": "2.0", "tolerance": 0.02,
        "steps_required": ["centre x and y", "b = sum((x-xb)(y-yb)) / sum((x-xb)^2)"],
        "solution": "x̄=1.5, ȳ=4. b = ((-0.5)(-1) + (0.5)(1)) / (0.25+0.25) = 1/0.5 = 2.",
        "explanation": "Intercept a = ȳ - b x̄ = 4 - 3 = 1, so the fit is exact on 2 points.",
    },
    {
        "topic": "ml.model_evaluation", "skills": ["ml.metrics"], "level": "beginner",
        "kind": "math", "difficulty": 2, "est_minutes": 8,
        "title": "F1 from counts",
        "statement": "TP=90, FP=10, FN=30 (as in the bank question). Compute F1 to 3 decimals.",
        "expected_value": "0.818", "tolerance": 0.002,
        "steps_required": ["precision", "recall", "harmonic mean"],
        "solution": "P=0.9, R=0.75, F1 = 2PR/(P+R) = 1.35/1.65 = 0.818.",
        "explanation": "F1 is the harmonic mean, so it punishes an imbalance between precision and recall.",
    },
    {
        "topic": "math.probability", "skills": ["prob.basics"], "level": "beginner",
        "kind": "math", "difficulty": 2, "est_minutes": 6,
        "title": "Bayes with a real base rate",
        "statement": "Recompute the seed.2.bayes posterior if prevalence rises to 10% (same test). Answer in %, 2 decimals.",
        "expected_value": "68.75", "tolerance": 1.0,
        "steps_required": ["numerator 0.10*0.99", "denominator + 0.90*0.05"],
        "solution": "0.099 / (0.099 + 0.045) = 0.6875 -> 68.75% (0.05 tolerance covers rounding of the FP term).",
        "explanation": "Same test, 10x prevalence -> ~4x the positive predictive value. Base rates dominate.",
    },
    {
        "topic": "math.info_theory", "skills": ["info.entropy"], "level": "intermediate",
        "kind": "math", "difficulty": 3, "est_minutes": 12,
        "title": "Cross-entropy of a two-class softmax",
        "statement": "True label is class 1. Predicted probabilities p = [0.7, 0.3]. Compute cross-entropy loss in nats (natural log) to 4 decimals.",
        "expected_value": "1.2040", "tolerance": 0.001,
        "steps_required": ["-ln(0.3)"],
        "solution": "-ln(0.3) = 1.20397.",
        "explanation": "Compare with bits: -log2(0.3) = 1.737. Always state the log base when quoting a loss.",
    },
    {
        "topic": "nlp.transformers", "skills": ["nlp.attention"], "level": "advanced",
        "kind": "math", "difficulty": 4, "est_minutes": 15,
        "title": "Attention output by hand (2 tokens)",
        "statement": "q1 = [1, 0], k1 = [1, 0], k2 = [0, 1], v1 = [1, 2], v2 = [3, 4], d_k = 2. "
        "Compute the attention weights for token 1 using scores = (q.k)/sqrt(d_k) and softmax. "
        "Report the first weight to 3 decimals.",
        "expected_value": "0.670", "tolerance": 0.01,
        "steps_required": ["scores = q.k / sqrt(d_k)", "softmax over the 2 scores", "weighted sum of v"],
        "solution": "s1 = 1/sqrt(2) = 0.7071, s2 = 0. exp(0.7071) = 2.0281, exp(0) = 1. "
        "w1 = 2.0281 / 3.0281 = 0.6698, w2 = 0.3302. Output = 0.6698*[1,2] + 0.3302*[3,4] = [1.6604, 2.6608].",
        "explanation": "Softmax never zeroes a token out completely, so the output is always a blend - the mechanism behind 'attention dilution' in long sequences.",
    },
    {
        "topic": "math.linear_algebra", "skills": ["linalg.decomposition"], "level": "intermediate",
        "kind": "math", "difficulty": 3, "est_minutes": 10,
        "title": "Eigenvalues of a 2x2",
        "statement": "M = [[2, 1], [1, 2]]. Give the larger eigenvalue.",
        "expected_value": "3.0", "tolerance": 0.01,
        "steps_required": ["det(M - λI) = 0", "(2-λ)^2 - 1 = 0"],
        "solution": "λ = 3 and 1; eigenvector for 3 is [1, 1]/sqrt(2).",
        "explanation": "Symmetric matrices give orthogonal eigenvectors - the reason PCA components are uncorrelated.",
    },
    {
        "topic": "adv.tuning", "skills": ["adv.nested_cv", "adv.tuning_skills"], "level": "intermediate",
        "kind": "math", "difficulty": 3, "est_minutes": 10,
        "title": "Search budget arithmetic",
        "statement": "5-fold CV, 60 configs, each fit 8s, each row of X costs 0.4ms to preprocess (n=200k, preprocessing refit per fold-config). Estimate total GPU-seconds for preprocessing only (in hours, 1 decimal).",
        "expected_value": "6.7", "tolerance": 1.5,
        "steps_required": ["folds*configs", "n * ms", "convert"],
        "solution": "300 fits * 200k * 0.4ms = 300 * 80s = 24000s = 6.7h.",
        "explanation": "Preprocessing can dwarf training; cache the transformed folds (or use a pipeline with memory-mapped artefacts).",
    },
]


# --------------------------------------------------------------------------- #
#  Coding tasks (real tests)
# --------------------------------------------------------------------------- #
def CT(slug, title, description, *, difficulty=2, level="beginner", topic="", skills=None, libraries=None,
       from_scratch=False, starter="", tests="", hints=None, solution="", explanation="", banned=None, est=20,
       xp=20) -> dict[str, Any]:
    return {
        "slug": slug, "title": title, "description": description, "difficulty": difficulty, "level": level,
        "topic": topic, "skills": skills or [], "libraries": libraries or ["numpy"], "from_scratch": from_scratch,
        "starter_code": starter, "test_code": tests, "hints": hints or [], "solution": solution,
        "solution_explanation": explanation, "banned_imports": banned or [], "est_minutes": est, "xp": xp,
    }


CODING_TASKS: list[dict[str, Any]] = [
    CT("numpy.moving_average", "Moving average with NumPy",
       "Implement `moving_average(x, k)` returning the mean of every window of length `k` over a 1-D array, "
       "'valid' mode: result length is len(x) - k + 1. No Python-level loops over the window.",
       topic="python.numpy", skills=["np.broadcast", "np.arrays"], libraries=["numpy"], difficulty=2,
       starter="import numpy as np\n\n\ndef moving_average(x, k: int):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import moving_average\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_small(self):\n"
             "        self.assertTrue(np.allclose(moving_average(np.array([1.,2.,3.,4.]), 2), [1.5, 2.5, 3.5]))\n"
             "    def test_matches_loop(self):\n"
             "        rng = np.random.default_rng(0)\n"
             "        x = rng.normal(size=257)\n"
             "        for k in (1, 3, 7, 16):\n"
             "            ref = np.array([x[i:i+k].mean() for i in range(len(x) - k + 1)])\n"
             "            self.assertTrue(np.allclose(moving_average(x, k), ref), msg='k=' + str(k))\n"
             "    def test_k_equals_len(self):\n"
             "        self.assertAlmostEqual(float(moving_average(np.array([1., 2., 3.]), 3)), 2.0)\n"
             "    def test_no_loop_over_window(self):\n"
             "        import inspect\n"
             "        src = inspect.getsource(moving_average)\n"
             "        self.assertNotIn('for i in range', src)\n",
       hints=["cumsum trick: c = np.concatenate([[0], np.cumsum(x)]); out = (c[k:] - c[:-k]) / k",
              "or use sliding_window_view from numpy.lib.stride_tricks"],
       solution="import numpy as np\n\n\ndef moving_average(x, k):\n    x = np.asarray(x, dtype=float)\n    c = np.concatenate([[0.0], np.cumsum(x)])\n    return (c[k:] - c[:-k]) / k\n",
       explanation="The cumulative-sum form is O(n) with one vectorised pass; np.convolve(x, np.ones(k)/k, 'valid') is the one-liner equivalent."),

    CT("math.mean_variance", "mean and variance from scratch",
       "Implement `mean(x)` and `variance(x)` (population, i.e. divide by n) for a 1-D array/list. "
       "Do not use numpy's `mean`/`var` - the point is to know what they compute.",
       from_scratch=True, topic="python.statistics", skills=["stat.describe", "np.arrays"], libraries=["numpy"], difficulty=1,
       starter="import numpy as np\n\n\ndef mean(x):\n    raise NotImplementedError\n\n\ndef variance(x):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import mean, variance\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_basic(self):\n"
             "        self.assertAlmostEqual(mean([2, 4, 4, 4, 5, 5, 7, 9]), 5.0)\n"
             "        self.assertAlmostEqual(variance([2, 4, 4, 4, 5, 5, 7, 9]), 4.0)\n"
             "    def test_random(self):\n"
             "        rng = np.random.default_rng(7)\n"
             "        x = rng.normal(3, 2, size=1000)\n"
             "        self.assertAlmostEqual(mean(x), float(np.mean(x)), places=10)\n"
             "        self.assertAlmostEqual(variance(x), float(np.var(x)), places=10)\n"
             "    def test_single(self):\n"
             "        self.assertAlmostEqual(variance([3.5]), 0.0)\n"
             "    def test_no_np_mean(self):\n"
             "        import inspect\n"
             "        src = inspect.getsource(mean) + inspect.getsource(variance)\n"
             "        self.assertNotIn('np.mean', src)\n"
             "        self.assertNotIn('np.var', src)\n",
       hints=["variance = mean of squared deviations from the mean", "convert input with np.asarray(x, dtype=float)"],
       solution="import numpy as np\n\n\ndef mean(x):\n    x = np.asarray(x, dtype=float)\n    return float(x.sum() / x.size)\n\n\ndef variance(x):\n    x = np.asarray(x, dtype=float)\n    mu = x.sum() / x.size\n    d = x - mu\n    return float((d * d).sum() / x.size)\n",
       explanation="Two-pass form above is numerically stable; the single-pass E[x^2]-E[x]^2 form loses precision when the mean is large relative to the spread."),

    CT("linalg.dot_matmul", "Dot product and matrix multiply from scratch",
       "Implement `dot(a, b)` and `matmul(A, B)` for 1-D/2-D float arrays without using `np.dot`, `@`, `np.matmul` or `np.einsum`. "
       "matmul must raise ValueError with a message containing 'shape' on inner-dimension mismatch.",
       from_scratch=True, topic="math.linear_algebra", skills=["linalg.matrices", "linalg.vectors", "np.linalg_ops"],
       libraries=["numpy"], difficulty=2,
       starter="import numpy as np\n\n\ndef dot(a, b):\n    raise NotImplementedError\n\n\ndef matmul(A, B):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import dot, matmul\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_dot(self):\n"
             "        self.assertEqual(dot(np.array([1., 2., 3.]), np.array([4., 0., 2.])), 10.0)\n"
             "    def test_matmul(self):\n"
             "        rng = np.random.default_rng(1)\n"
             "        A = rng.normal(size=(7, 5))\n"
             "        B = rng.normal(size=(5, 4))\n"
             "        self.assertTrue(np.allclose(matmul(A, B), A @ B))\n"
             "    def test_batch(self):\n"
             "        rng = np.random.default_rng(2)\n"
             "        for _ in range(20):\n"
             "            m, k, n = rng.integers(1, 9, size=3)\n"
             "            A = rng.normal(size=(int(m), int(k)))\n"
             "            B = rng.normal(size=(int(k), int(n)))\n"
             "            self.assertTrue(np.allclose(matmul(A, B), A @ B, atol=1e-8))\n"
             "    def test_shape_error(self):\n"
             "        with self.assertRaises(ValueError) as ctx:\n"
             "            matmul(np.ones((3, 4)), np.ones((5, 2)))\n"
             "        self.assertIn('shape', str(ctx.exception).lower())\n"
             "    def test_no_dot(self):\n"
             "        import inspect\n"
             "        src = inspect.getsource(matmul) + inspect.getsource(dot)\n"
             "        for bad in ('np.dot', '@', 'np.matmul', 'einsum'):\n"
             "            self.assertNotIn(bad, src)\n",
       hints=["A[i, :] * B[:, j] summed - but vectorise over j: (A[:, :, None] * B[None, :, :]).sum(axis=1)",
              "check A.shape[1] == B.shape[0] first"],
       solution="import numpy as np\n\n\ndef dot(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    if a.shape != b.shape:\n        raise ValueError('shape mismatch in dot')\n    return float(np.sum(a * b))\n\n\ndef matmul(A, B):\n    A = np.asarray(A, dtype=float)\n    B = np.asarray(B, dtype=float)\n    if A.ndim != 2 or B.ndim != 2:\n        raise ValueError('matmul expects 2-D arrays')\n    if A.shape[1] != B.shape[0]:\n        raise ValueError('shape mismatch: inner dims must agree')\n    return (A[:, :, None] * B[None, :, :]).sum(axis=1)\n",
       explanation="Broadcasting to a (m, n, k) temporary is how you write matmul without calling matmul; it is memory-heavy (m*n*k), which is exactly why real BLAS kernels tile the product."),

    CT("ml.linear_regression_gd", "Linear regression by gradient descent",
       "Implement `fit_linear_regression(X, y, lr=0.1, iters=1000)` returning weights `w` (d,) and intercept `b`. "
       "Full-batch gradient descent on MSE with the normalised design matrix handled internally (standardise X before the updates, then map back or predict consistently - provide `predict(X, w, b)` too).",
       from_scratch=True, topic="ml.linear_models", skills=["ml.linear_regression_impl", "ml.gradient_descent_impl", "np.linalg_ops"],
       libraries=["numpy"], difficulty=3,
       starter="import numpy as np\n\n\ndef fit_linear_regression(X, y, lr=0.1, iters=1000):\n    raise NotImplementedError\n\n\ndef predict(X, w, b):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import fit_linear_regression, predict\n\n"
             "def make():\n"
             "    rng = np.random.default_rng(0)\n"
             "    X = rng.normal(size=(200, 3))\n"
             "    true = np.array([2.0, -1.5, 0.5])\n"
             "    y = X @ true + 3.0 + rng.normal(scale=0.05, size=200)\n"
             "    return X, y, true\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_recovers(self):\n"
             "        X, y, true = make()\n"
             "        w, b = fit_linear_regression(X, y, lr=0.1, iters=4000)\n"
             "        self.assertTrue(np.allclose(np.asarray(w), true, atol=0.05), msg=str(w))\n"
             "        self.assertAlmostEqual(float(b), 3.0, delta=0.05)\n"
             "    def test_vs_lstsq(self):\n"
             "        X, y, _ = make()\n"
             "        w, b = fit_linear_regression(X, y, lr=0.1, iters=6000)\n"
             "        pred = np.asarray(predict(X, w, b), dtype=float)\n"
             "        A = np.hstack([X, np.ones((len(X), 1))])\n"
             "        sol, *_ = np.linalg.lstsq(A, y, rcond=None)\n"
             "        self.assertLess(float(np.mean((pred - A @ sol) ** 2)), 1e-3)\n"
             "    def test_deterministic(self):\n"
             "        X, y, _ = make()\n"
             "        w1, _ = fit_linear_regression(X, y, lr=0.1, iters=800)\n"
             "        w2, _ = fit_linear_regression(X, y, lr=0.1, iters=800)\n"
             "        self.assertTrue(np.allclose(w1, w2))\n",
       hints=["MSE gradient: (2/n) * X^T (Xw + b - y)", "standardise columns, fit, then un-standardise the coefficients",
              "add an intercept column of ones instead of tracking b separately (then b = w[-1])"],
       solution="import numpy as np\n\n\ndef fit_linear_regression(X, y, lr=0.1, iters=1000):\n    X = np.asarray(X, dtype=float)\n    y = np.asarray(y, dtype=float)\n    mu, sd = X.mean(0), X.std(0)\n    sd = np.where(sd == 0, 1.0, sd)\n    Z = (X - mu) / sd\n    A = np.hstack([Z, np.ones((len(Z), 1))])\n    ymu = y.mean()\n    t = y - ymu\n    w = np.zeros(A.shape[1])\n    n = len(A)\n    for _ in range(int(iters)):\n        grad = (2.0 / n) * (A.T @ (A @ w - t))\n        w -= lr * grad\n    coef = w[:-1] / sd\n    intercept = float(ymu + t.mean() - (coef * mu).sum())\n    return coef, intercept\n\n\ndef predict(X, w, b):\n    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + float(b)\n",
       explanation="Standardising makes the Hessian well-conditioned, so plain GD with lr=0.1 converges; on raw data you would need a much smaller step."),

    CT("ml.logistic_from_scratch", "Logistic regression with gradient descent",
       "Implement `class LogisticRegression` with `fit(X, y, lr, iters, l2)`, `predict_proba(X)`, `predict(X, threshold=0.5)`. "
       "No sklearn. Must be numerically stable (clip or use log-sum-exp for the loss).",
       from_scratch=True, topic="ml.linear_models", skills=["ml.logistic_regression", "calc.partial_gradient", "np.linalg_ops"],
       libraries=["numpy"], difficulty=3,
       starter="import numpy as np\n\n\nclass LogisticRegression:\n    def fit(self, X, y, lr=0.3, iters=1500, l2=0.0):\n        raise NotImplementedError\n\n    def predict_proba(self, X):\n        raise NotImplementedError\n\n    def predict(self, X, threshold=0.5):\n        raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import LogisticRegression\n\n"
             "def make():\n"
             "    rng = np.random.default_rng(3)\n"
             "    X = np.vstack([rng.normal(-1.5, 1.0, size=(150, 2)), rng.normal(1.5, 1.0, size=(150, 2))])\n"
             "    y = np.array([0] * 150 + [1] * 150)\n"
             "    return X, y\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_separable(self):\n"
             "        X, y = make()\n"
             "        m = LogisticRegression().fit(X, y, lr=0.3, iters=2000)\n"
             "        acc = float((m.predict(X) == y).mean())\n"
             "        self.assertGreater(acc, 0.9)\n"
             "    def test_proba_shape_range(self):\n"
             "        X, y = make()\n"
             "        m = LogisticRegression().fit(X, y, lr=0.2, iters=500)\n"
             "        p = np.asarray(m.predict_proba(X))\n"
             "        self.assertEqual(p.shape, (len(X), 2))\n"
             "        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-6))\n"
             "        self.assertTrue((p >= 0).all() and (p <= 1).all())\n"
             "    def test_l2_shrinks(self):\n"
             "        X, y = make()\n"
             "        a = LogisticRegression().fit(X, y, lr=0.3, iters=1500, l2=0.0)\n"
             "        b = LogisticRegression().fit(X, y, lr=0.3, iters=1500, l2=1.0)\n"
             "        self.assertLess(float(np.linalg.norm(b.w)), float(np.linalg.norm(a.w)))\n"
             "    def test_stability(self):\n"
             "        X = np.array([[1e4], [1e4 + 1], [-1e4]])\n"
             "        y = np.array([1, 1, 0])\n"
             "        m = LogisticRegression().fit(X, y, lr=0.05, iters=200)\n"
             "        self.assertTrue(np.isfinite(m.predict_proba(X)).all())\n",
       hints=["sigmoid via np.where(z>=0, 1/(1+exp(-z)), exp(z)/(1+exp(z))) avoids overflow",
              "grad_w = X^T (p - y)/n + l2*w ; grad_b = mean(p - y)"],
       solution="import numpy as np\n\n\ndef _sigmoid(z):\n    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))\n\n\nclass LogisticRegression:\n    def fit(self, X, y, lr=0.3, iters=1500, l2=0.0):\n        X = np.asarray(X, float)\n        y = np.asarray(y, float).ravel()\n        self.mu, self.sd = X.mean(0), X.std(0)\n        self.sd = np.where(self.sd == 0, 1.0, self.sd)\n        Z = (X - self.mu) / self.sd\n        self.w = np.zeros(Z.shape[1])\n        self.b = 0.0\n        n = len(Z)\n        for _ in range(int(iters)):\n            p = _sigmoid(Z @ self.w + self.b)\n            gw = Z.T @ (p - y) / n + l2 * self.w\n            gb = float((p - y).mean())\n            self.w -= lr * gw\n            self.b -= lr * gb\n        self.w_raw = self.w / self.sd\n        self.b_raw = self.b - float((self.w * self.mu / self.sd).sum())\n        return self\n\n    def predict_proba(self, X):\n        z = np.asarray(X, float) @ self.w_raw + self.b_raw\n        p = _sigmoid(z)\n        return np.vstack([1 - p, p]).T\n\n    def predict(self, X, threshold=0.5):\n        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)\n",
       explanation="Standardising + a numerically safe sigmoid is the whole difference between a working and a NaN-producing implementation. Raw-space coefficients are kept so `predict` works on unstandardised data."),

    CT("ml.kmeans_from_scratch", "k-means from scratch",
       "Implement `kmeans(X, k, iters=100, seed=0) -> (labels, centers)` using k-means++ init. "
       "Empty clusters must be re-seeded rather than producing NaN.",
       from_scratch=True, topic="ml.clustering", skills=["ml.clustering_kmeans", "np.broadcast"],
       libraries=["numpy"], difficulty=3,
       starter="import numpy as np\n\n\ndef kmeans(X, k=3, iters=100, seed=0):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import kmeans\n\n"
             "def blobs():\n"
             "    rng = np.random.default_rng(5)\n"
             "    parts = [rng.normal(loc, 0.4, size=(120, 2)) for loc in ([0, 0], [8, 1], [4, 9])]\n"
             "    X = np.vstack(parts)\n"
             "    truth = np.array([0] * 120 + [1] * 120 + [2] * 120)\n"
             "    return X, truth\n\n"
             "def purity(labels, truth):\n"
             "    from collections import Counter\n"
             "    pairs = Counter(zip(labels.tolist(), truth.tolist()))\n"
             "    by_label = Counter(labels.tolist())\n"
             "    return sum(max(pairs[l, t] for t in set(truth.tolist()) if (l, t) in pairs) for l in by_label) / len(truth)\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_recovers_blobs(self):\n"
             "        X, truth = blobs()\n"
             "        labels, centers = kmeans(X, 3, iters=100, seed=0)\n"
             "        self.assertEqual(centers.shape, (3, 2))\n"
             "        self.assertGreater(purity(np.asarray(labels), truth), 0.97)\n"
             "    def test_deterministic(self):\n"
             "        X, _ = blobs()\n"
             "        a, ca = kmeans(X, 3, iters=40, seed=1)\n"
             "        b, cb = kmeans(X, 3, iters=40, seed=1)\n"
             "        self.assertTrue(np.allclose(np.asarray(a), np.asarray(b)))\n"
             "        self.assertTrue(np.allclose(ca, cb))\n"
             "    def test_no_nan(self):\n"
             "        X = np.array([[0., 0.], [0., 0.], [5., 5.], [5., 5.]])\n"
             "        labels, centers = kmeans(X, 4, iters=25, seed=2)\n"
             "        self.assertTrue(np.isfinite(np.asarray(centers)).all())\n"
             "        self.assertEqual(len(np.asarray(labels)), 4)\n",
       hints=["distance to all centers: ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)",
              "k-means++: pick next center with probability proportional to current min-distance^2",
              "np.bincount(labels, weights=X[:, d], minlength=k) / counts gives the new centers"],
       solution="import numpy as np\n\n\ndef kmeans(X, k=3, iters=100, seed=0):\n    X = np.asarray(X, float)\n    n = len(X)\n    rng = np.random.default_rng(seed)\n    centers = np.empty((k, X.shape[1]), float)\n    centers[0] = X[rng.integers(n)]\n    for j in range(1, k):\n        d = ((X[:, None, :] - centers[None, :j]) ** 2).sum(-1).min(1)\n        probs = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)\n        centers[j] = X[rng.choice(n, p=probs)]\n    labels = np.zeros(n, dtype=int)\n    for _ in range(int(iters)):\n        dist = ((X[:, None, :] - centers[None, :]) ** 2).sum(-1)\n        new_labels = dist.argmin(1)\n        counts = np.bincount(new_labels, minlength=k).astype(float)\n        for dim in range(X.shape[1]):\n            sums = np.bincount(new_labels, weights=X[:, dim], minlength=k)\n            upd = np.divide(sums, counts, out=np.zeros(k), where=counts > 0)\n            empty = counts == 0\n            if empty.any():\n                idx = rng.choice(n, size=int(empty.sum()), replace=False)\n                upd[empty] = X[idx, dim]\n            centers[empty | ~empty, dim] = upd\n        if np.array_equal(new_labels, labels) and _ > 0:\n            break\n        labels = new_labels\n    return labels, centers\n",
       explanation="Assignment + recomputation per dimension via bincount keeps it vectorised (no Python loop over points). Re-seeding empty clusters avoids the NaN failure that plain Lloyd's has on duplicate data."),

    CT("nn.softmax_crossentropy_grad", "Stable softmax + cross-entropy and its gradient",
       "Implement `softmax(z)` (row-wise, overflow-safe), `cross_entropy(logits, y)` returning the mean loss, and "
       "`d_logits(logits, y)` returning dL/dlogits. The gradient must equal (softmax(z) - onehot(y)) / n.",
       from_scratch=True, topic="math.info_theory", skills=["info.entropy", "ml.logistic_regression", "dl.backprop"],
       libraries=["numpy"], difficulty=3,
       starter="import numpy as np\n\n\ndef softmax(z):\n    raise NotImplementedError\n\n\ndef cross_entropy(logits, y):\n    raise NotImplementedError\n\n\ndef d_logits(logits, y):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import softmax, cross_entropy, d_logits\n\n"
             "class T(unittest.TestCase):\n"
             "    def setUp(self):\n"
             "        rng = np.random.default_rng(11)\n"
             "        self.z = rng.normal(size=(64, 5))\n"
             "        self.y = rng.integers(0, 5, size=64)\n"
             "    def test_softmax_rows(self):\n"
             "        p = softmax(self.z)\n"
             "        self.assertTrue(np.allclose(p.sum(1), 1.0))\n"
             "        self.assertTrue((p > 0).all())\n"
             "    def test_softmax_stable(self):\n"
             "        big = np.array([[1000.0, 1001.0, 999.0]])\n"
             "        p = softmax(big)\n"
             "        self.assertTrue(np.isfinite(p).all())\n"
             "        self.assertAlmostEqual(float(p[0].sum()), 1.0, places=10)\n"
             "    def test_loss_matches_manual(self):\n"
             "        p = softmax(self.z)\n"
             "        manual = -np.mean(np.log(p[np.arange(len(p)), self.y]))\n"
             "        self.assertAlmostEqual(float(cross_entropy(self.z, self.y)), float(manual), places=8)\n"
             "    def test_gradient(self):\n"
             "        g = d_logits(self.z, self.y)\n"
             "        onehot = np.zeros_like(self.z)\n"
             "        onehot[np.arange(len(self.y)), self.y] = 1.0\n"
             "        expected = (softmax(self.z) - onehot) / len(self.y)\n"
             "        self.assertTrue(np.allclose(g, expected, atol=1e-8))\n"
             "    def test_numeric_gradient(self):\n"
             "        z = self.z[:8].copy()\n"
             "        y = self.y[:8]\n"
             "        eps = 1e-6\n"
             "        num = np.zeros_like(z)\n"
             "        for i in range(2):\n"
             "            for j in range(3):\n"
             "                zp = z.copy(); zp[i, j] += eps\n"
             "                zm = z.copy(); zm[i, j] -= eps\n"
             "                num[i, j] = (cross_entropy(zp, y) - cross_entropy(zm, y)) / (2 * eps)\n"
             "        self.assertTrue(np.allclose(d_logits(z, y)[:2, :3], num[:2, :3], atol=1e-5))\n",
       hints=["subtract the row max before exp", "clip probabilities to [eps, 1] before log, or use log-softmax directly",
              "the gradient of the mean CE w.r.t. logits is (p - onehot)/n"],
       solution="import numpy as np\n\n\ndef softmax(z):\n    z = np.asarray(z, float)\n    s = z - z.max(axis=-1, keepdims=True)\n    e = np.exp(s)\n    return e / e.sum(axis=-1, keepdims=True)\n\n\ndef _log_probs(logits):\n    z = np.asarray(logits, float)\n    shift = z - z.max(axis=-1, keepdims=True)\n    return shift - np.log(np.exp(shift).sum(axis=-1, keepdims=True))\n\n\ndef cross_entropy(logits, y):\n    y = np.asarray(y, int)\n    lp = _log_probs(logits)[np.arange(len(y)), y]\n    return float(-lp.mean())\n\n\ndef d_logits(logits, y):\n    y = np.asarray(y, int)\n    p = softmax(logits)\n    grad = p - np.eye(p.shape[1])[y]\n    return grad / p.shape[0]\n",
       explanation="log_softmax keeps the loss finite for huge logits; the (p - onehot)/n gradient is the reason CE + softmax is used instead of MSE."),

    CT("nn.backprop_two_layer", "Two-layer network: forward and backward",
       "Implement `forward(x, W1, b1, W2, b2) -> (loss, cache)` for a single hidden ReLU layer trained with MSE on target `t`, "
       "and `backward(cache) -> (dW1, db1, dW2, db2)`. Gradients must match finite differences.",
       from_scratch=True, topic="dl.backprop", skills=["dl.backprop", "dl.backprop_impl", "calc.partial_gradient"],
       libraries=["numpy"], difficulty=4, est=40, xp=40,
       starter="import numpy as np\n\n\ndef forward(x, t, W1, b1, W2, b2):\n    raise NotImplementedError\n\n\ndef backward(cache):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import forward, backward\n\n"
             "def setup():\n"
             "    rng = np.random.default_rng(4)\n"
             "    x = rng.normal(size=(5, 4))\n"
             "    t = rng.normal(size=(5, 2))\n"
             "    W1 = rng.normal(scale=0.5, size=(4, 6))\n"
             "    b1 = np.zeros(6)\n"
             "    W2 = rng.normal(scale=0.5, size=(6, 2))\n"
             "    b2 = np.zeros(2)\n"
             "    return x, t, W1, b1, W2, b2\n\n"
             "def loss_of(x, t, W1, b1, W2, b2):\n"
             "    h = np.maximum(0, x @ W1 + b1)\n"
             "    y = h @ W2 + b2\n"
             "    return float(((y - t) ** 2).mean())\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_forward_matches(self):\n"
             "        x, t, W1, b1, W2, b2 = setup()\n"
             "        loss, cache = forward(x, t, W1, b1, W2, b2)\n"
             "        self.assertAlmostEqual(float(loss), loss_of(x, t, W1, b1, W2, b2), places=9)\n"
             "    def test_backward_finite_diff(self):\n"
             "        x, t, W1, b1, W2, b2 = setup()\n"
             "        _loss, cache = forward(x, t, W1, b1, W2, b2)\n"
             "        dW1, db1, dW2, db2 = backward(cache)\n"
             "        eps = 1e-6\n"
             "        for name, arr, grad in (('W1', W1, dW1), ('b1', b1, db1), ('W2', W2, dW2), ('b2', b2, db2)):\n"
             "            num = np.zeros_like(arr, dtype=float)\n"
             "            flat_idx = np.ndindex(arr.shape)\n"
             "            for idx in flat_idx:\n"
             "                up = arr.copy(); up[idx] += eps\n"
             "                dn = arr.copy(); dn[idx] -= eps\n"
             "                args = {'W1': up if name == 'W1' else W1, 'b1': up if name == 'b1' else b1,\n"
             "                        'W2': up if name == 'W2' else W2, 'b2': up if name == 'b2' else b2}\n"
             "                dns = {'W1': dn if name == 'W1' else W1, 'b1': dn if name == 'b1' else b1,\n"
             "                       'W2': dn if name == 'W2' else W2, 'b2': dn if name == 'b2' else b2}\n"
             "                num[idx] = (loss_of(x, t, args['W1'], args['b1'], args['W2'], args['b2'])\n"
             "                            - loss_of(x, t, dns['W1'], dns['b1'], dns['W2'], dns['b2'])) / (2 * eps)\n"
             "            self.assertTrue(np.allclose(np.asarray(grad), num, atol=1e-5), msg=name)\n"
             "    def test_shapes(self):\n"
             "        x, t, W1, b1, W2, b2 = setup()\n"
             "        _l, cache = forward(x, t, W1, b1, W2, b2)\n"
             "        dW1, db1, dW2, db2 = backward(cache)\n"
             "        self.assertEqual(np.asarray(dW1).shape, W1.shape)\n"
             "        self.assertEqual(np.asarray(db1).shape, b1.shape)\n"
             "        self.assertEqual(np.asarray(dW2).shape, W2.shape)\n"
             "        self.assertEqual(np.asarray(db2).shape, b2.shape)\n",
       hints=["cache = (x, z1, h, y, t) ... keep everything needed for the local derivatives",
              "dy = 2*(y - t)/n ; dW2 = h^T dy ; dh = dy @ W2^T ; dz1 = dh * (z1 > 0)",
              "note z1 is the pre-activation: the ReLU mask must come from z1, not h"],
       solution="import numpy as np\n\n\ndef forward(x, t, W1, b1, W2, b2):\n    x, t = np.asarray(x, float), np.asarray(t, float)\n    z1 = x @ W1 + b1\n    h = np.maximum(0, z1)\n    y = h @ W2 + b2\n    loss = float(((y - t) ** 2).mean())\n    return loss, (x, t, W1, b1, W2, b2, z1, h, y)\n\n\ndef backward(cache):\n    x, t, W1, b1, W2, b2, z1, h, y = cache\n    # the loss is the mean over ALL (n, C) entries, so the constant is 2 / y.size\n    dy = 2.0 * (y - t) / y.size\n    dW2 = h.T @ dy\n    db2 = dy.sum(0)\n    dh = dy @ W2.T\n    dz1 = dh * (z1 > 0)\n    dW1 = x.T @ dz1\n    db1 = dz1.sum(0)\n    return dW1, db1, dW2, db2\n",
       explanation="This is reverse-mode autograd on a 3-node graph. The constant in dL/dy depends on how you normalise the loss (mean over n, or over n*C) - which is exactly the mistake a finite-difference check catches. Always validate a hand-written backward pass that way."),

    CT("pandas.group_metrics", "Group metrics, ties and sorting in Pandas",
       "Implement `summary(df)` that returns, for each `team`, one row with: `games` (count), `points` (sum of `pts`), "
       "`avg_ppg` (mean of `ppg`), and `top_scorer` (name of the max `pts` for that team; ties -> alphabetically first). "
       "Result must be sorted by points desc, then team asc, with the index reset.",
       topic="python.pandas", skills=["pd.dataframes", "pd.cleaning"], libraries=["pandas", "numpy"], difficulty=2,
       starter="import pandas as pd\n\n\ndef summary(df):\n    raise NotImplementedError\n",
       tests="import unittest\nimport pandas as pd\nimport numpy as np\nfrom solution import summary\n\n"
             "def frame():\n"
             "    df = pd.DataFrame({\n"
             "        'team': ['A', 'A', 'B', 'B', 'B', 'C'],\n"
             "        'name': ['ana', 'bo', 'cy', 'di', 'ed', 'fi'],\n"
             "        'pts': [10, 10, 4, 9, 9, 7],\n"
             "        'ppg': [2.0, 4.0, 1.0, 3.0, 3.0, 7.0],\n"
             "    })\n"
             "    return df\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_values(self):\n"
             "        out = summary(frame())\n"
             "        self.assertEqual(list(out.columns[:4]), ['team', 'games', 'points', 'avg_ppg'])\n"
             "        self.assertEqual(list(out['team']), ['B', 'A', 'C'])\n"
             "        self.assertEqual(list(out['points']), [22, 20, 7])\n"
             "        self.assertEqual(list(out['games']), [3, 2, 1])\n"
             "        self.assertTrue(np.allclose(out['avg_ppg'], [2.3333333333, 3.0, 7.0], atol=1e-6))\n"
             "    def test_tie_break(self):\n"
             "        out = summary(frame())\n"
             "        self.assertEqual(list(out['top_scorer']), ['di', 'ana', 'fi'])\n"
             "    def test_empty(self):\n"
             "        empty = frame().iloc[0:0]\n"
             "        out = summary(empty)\n"
             "        self.assertEqual(len(out), 0)\n",
       hints=["sort by ['pts','name'] then use groupby(...).first() for the top scorer",
              "agg with named aggregation: games=('name','size'), points=('pts','sum')"],
       solution="import pandas as pd\n\n\ndef summary(df):\n    if len(df) == 0:\n        return pd.DataFrame(columns=['team', 'games', 'points', 'avg_ppg', 'top_scorer'])\n    ordered = df.sort_values(['pts', 'name'], ascending=[False, True])\n    top = ordered.groupby('team')['name'].first().rename('top_scorer')\n    out = df.groupby('team').agg(games=('name', 'size'), points=('pts', 'sum'), avg_ppg=('ppg', 'mean'))\n    out = out.join(top).reset_index()\n    return out.sort_values(['points', 'team'], ascending=[False, True]).reset_index(drop=True)\n",
       explanation="Deterministic tie-breaking needs an explicit secondary sort key before `first()`; groupby.apply for this is both slower and harder to reason about."),

    CT("sql.window_features", "SQL: window functions for ML features",
       "Using the sqlite3 stdlib, implement `build_features(rows)` that turns a list of dicts "
       "`{'user_id': int, 'amount': float, 'day': int}` into per-transaction features: cumulative spend for the user, "
       "transaction index (1-based) and share of the user's running total. Return a list of tuples "
       "(user_id, day, cum_amount, txn_index, share) ordered by user_id, day. Use a real SQL query, not a Python loop.",
       topic="python.sql", skills=["sql.advanced", "sql.select"], libraries=["sqlite3"], difficulty=3,
       starter="import sqlite3\n\n\ndef build_features(rows):\n    raise NotImplementedError\n",
       tests="import unittest\nfrom solution import build_features\n\n"
             "ROWS = [\n"
             "    {'user_id': 1, 'amount': 10.0, 'day': 1},\n"
             "    {'user_id': 1, 'amount': 5.0, 'day': 2},\n"
             "    {'user_id': 2, 'amount': 4.0, 'day': 1},\n"
             "    {'user_id': 1, 'amount': 5.0, 'day': 3},\n"
             "    {'user_id': 2, 'amount': 12.0, 'day': 4},\n"
             "]\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_shape_and_values(self):\n"
             "        out = build_features(ROWS)\n"
             "        self.assertEqual(len(out), 5)\n"
             "        first = out[0]\n"
             "        self.assertEqual(first[0], 1)\n"
             "        self.assertAlmostEqual(first[2], 10.0, places=6)\n"
             "        self.assertEqual(first[3], 1)\n"
             "        self.assertAlmostEqual(first[4], 1.0, places=6)\n"
             "    def test_user2_running(self):\n"
             "        out = build_features(ROWS)\n"
             "        u2 = [r for r in out if r[0] == 2]\n"
             "        self.assertAlmostEqual(u2[-1][2], 16.0, places=6)\n"
             "        self.assertEqual(u2[-1][3], 2)\n"
             "    def test_uses_sql(self):\n"
             "        import inspect\n"
             "        src = inspect.getsource(build_features).lower()\n"
             "        self.assertIn('over', src)\n"
             "        self.assertTrue('partition by' in src)\n",
       hints=["create a temp table, insert with executemany, then one SELECT with SUM() OVER (PARTITION BY user_id ORDER BY day ROWS UNBOUNDED PRECEDING)",
              "share needs the running total, not the grand total, per the spec above"],
       solution="import sqlite3\n\n\ndef build_features(rows):\n    con = sqlite3.connect(':memory:')\n    con.execute('CREATE TABLE t (user_id INTEGER, amount REAL, day INTEGER)')\n    con.executemany('INSERT INTO t VALUES (?, ?, ?)', [(r['user_id'], r['amount'], r['day']) for r in rows])\n    sql = '''\n        SELECT user_id, day,\n               SUM(amount) OVER (PARTITION BY user_id ORDER BY day ROWS UNBOUNDED PRECEDING) AS cum_amount,\n               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS txn_index,\n               amount * 1.0 / SUM(amount) OVER (PARTITION BY user_id ORDER BY day ROWS UNBOUNDED PRECEDING) AS share\n        FROM t\n        ORDER BY user_id, day\n    '''\n    out = con.execute(sql).fetchall()\n    con.close()\n    return out\n",
       explanation="Window aggregates compute per-row features without collapsing the grain - the same pattern you use for lag/rolling features in a warehouse."),

    CT("sklearn.pipeline_cv", "sklearn: a leakage-free pipeline",
       "Build `run(X, y) -> dict` that: creates a Pipeline with a StandardScaler + LogisticRegression, evaluates it with "
       "`cross_val_score(..., cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')`, and returns "
       "`{'mean': float, 'std': float, 'n_folds': int}`. Fitting a scaler outside the pipeline must NOT be used.",
       topic="ml.preparation", skills=["ml.pipelines", "adv.nested_cv"], libraries=["sklearn", "numpy"], difficulty=2,
       starter="from sklearn.linear_model import LogisticRegression\n\n\ndef run(X, y):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import run\n\n"
             "def data():\n"
             "    rng = np.random.default_rng(0)\n"
             "    y = rng.integers(0, 2, size=300)\n"
             "    X = rng.normal(size=(300, 4)) + y[:, None] * 0.9\n"
             "    return X, y\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_keys_and_range(self):\n"
             "        out = run(*data())\n"
             "        self.assertEqual({'mean', 'std', 'n_folds'} <= set(out.keys()), True)\n"
             "        self.assertEqual(out['n_folds'], 5)\n"
             "        self.assertTrue(0.5 < out['mean'] < 1.0)\n"
             "    def test_deterministic(self):\n"
             "        a = run(*data())\n"
             "        b = run(*data())\n"
             "        self.assertAlmostEqual(a['mean'], b['mean'], places=10)\n"
             "    def test_pipeline_used(self):\n"
             "        import inspect\n"
             "        src = inspect.getsource(run)\n"
             "        self.assertTrue('Pipeline' in src or 'make_pipeline' in src)\n"
             "        self.assertIn('cross_val_score', src)\n",
       hints=["from sklearn.pipeline import make_pipeline; from sklearn.model_selection import cross_val_score, StratifiedKFold",
              "scoring='roc_auc' with y_proba-based scorers works for LogisticRegression automatically"],
       solution="import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n\ndef run(X, y):\n    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\n    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')\n    return {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'n_folds': int(len(scores))}\n",
       explanation="make_pipeline guarantees the scaler is refit inside each fold - the difference between an honest AUC and an optimistic one."),

    CT("mlops.evaluator_threshold", "Production evaluator: threshold under constraints",
       "Implement `choose_threshold(y_true, proba, min_precision=0.2, min_recall=0.9)` returning the score threshold that "
       "maximises F1 among thresholds satisfying both floors (0.0 if none does, and `{'threshold','precision','recall','f1','ok'}`). "
       "Candidate thresholds: the unique predicted scores plus 0.0.",
       topic="adv.imbalanced", skills=["adv.cost_sensitive", "adv.imbalance", "ml.metrics"], libraries=["numpy"],
       difficulty=3, est=30, xp=30,
       starter="import numpy as np\n\n\ndef choose_threshold(y_true, proba, min_precision=0.2, min_recall=0.9):\n    raise NotImplementedError\n",
       tests="import unittest\nimport numpy as np\nfrom solution import choose_threshold\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_perfect_case(self):\n"
             "        y = np.array([0, 0, 1, 1, 1])\n"
             "        p = np.array([0.1, 0.2, 0.7, 0.8, 0.9])\n"
             "        out = choose_threshold(y, p, min_precision=0.9, min_recall=0.9)\n"
             "        self.assertTrue(out['ok'])\n"
             "        self.assertGreaterEqual(out['precision'], 0.9 - 1e-9)\n"
             "        self.assertGreaterEqual(out['recall'], 0.9 - 1e-9)\n"
             "    def test_impossible_constraint(self):\n"
             "        y = np.array([0, 1, 0, 1, 0])\n"
             "        p = np.array([0.9, 0.1, 0.8, 0.2, 0.7])\n"
             "        out = choose_threshold(y, p, min_precision=0.99, min_recall=0.99)\n"
             "        self.assertFalse(out['ok'])\n"
             "        self.assertEqual(out['threshold'], 0.0)\n"
             "    def test_recall_monotonic_choice(self):\n"
             "        rng = np.random.default_rng(2)\n"
             "        y = (rng.random(400) < 0.1).astype(int)\n"
             "        p = np.clip(rng.normal(0.1, 0.2, size=len(y)) + 0.4 * y, 0, 1)\n"
             "        out = choose_threshold(y, p, min_precision=0.2, min_recall=0.9)\n"
             "        if out['ok']:\n"
             "            self.assertGreaterEqual(out['recall'], 0.9 - 1e-9)\n"
             "            self.assertGreaterEqual(out['precision'], 0.2 - 1e-9)\n"
             "    def test_returns_floats(self):\n"
             "        y = np.array([0, 1]); p = np.array([0.4, 0.6])\n"
             "        out = choose_threshold(y, p, 0.1, 0.1)\n"
             "        for k in ('threshold', 'precision', 'recall', 'f1'):\n"
             "            self.assertIsInstance(out[k], float)\n",
       hints=["vectorise over candidate thresholds: preds = (p[:, None] >= t[None, :])",
              "recall = tp / (tp + fn); guard division by zero when there are no positives",
              "tie-break: prefer the higher threshold (fewer alerts) at equal F1"],
       solution="import numpy as np\n\n\ndef choose_threshold(y_true, proba, min_precision=0.2, min_recall=0.9):\n    y = np.asarray(y_true).astype(int).ravel()\n    p = np.asarray(proba, dtype=float).ravel()\n    cands = np.unique(np.concatenate([[0.0], p]))\n    pos = y.sum()\n    best = None\n    for t in cands:\n        pred = (p >= t).astype(int)\n        tp = float(((pred == 1) & (y == 1)).sum())\n        fp = float(((pred == 1) & (y == 0)).sum())\n        fn = float(pos - tp)\n        precision = tp / (tp + fp) if (tp + fp) else 0.0\n        recall = tp / pos if pos else 0.0\n        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0\n        ok = precision >= min_precision - 1e-12 and recall >= min_recall - 1e-12\n        cand = {'threshold': float(t), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'ok': bool(ok)}\n        key = (1 if ok else 0, f1 if ok else -1.0, float(t))\n        if best is None or key > best[0]:\n            best = (key, cand)\n    result = best[1] if best else {'threshold': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ok': False}\n    if not result['ok']:\n        return {'threshold': 0.0, 'precision': result['precision'], 'recall': result['recall'], 'f1': result['f1'], 'ok': False}\n    return result\n",
       explanation="This is the real job: the model is fixed, and you choose the operating point under precision/recall floors. Notice the explicit tie-break - production code must be deterministic."),

    CT("debug.leakage_split", "Debug: fix the leaky split",
       "The function below leaks: the same `account_id` appears in both splits and the scaler is fit on all rows. "
       "Rewrite it so that (a) every account is in exactly one split and (b) the scaler is fit on the training rows only. "
       "Return `(train, test, scaler)`; you must keep `GroupShuffleSplit` semantics without sklearn's splitter being mandatory.",
       topic="ml.preparation", skills=["ml.data_splits", "ml.pipelines"], libraries=["pandas", "numpy", "sklearn"],
       difficulty=3, level="intermediate", est=25, xp=30,
       starter="import numpy as np\nimport pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n\n\ndef prepare(df):\n    # BUGGY on purpose: fix it\n    scaler = StandardScaler().fit(df[['x1', 'x2']])\n    train = df.iloc[: int(len(df) * 0.8)]\n    test = df.iloc[int(len(df) * 0.8):]\n    return train, test, scaler\n",
       tests="import unittest\nimport numpy as np\nimport pandas as pd\nfrom solution import prepare\n\n"
             "def frame():\n"
             "    rng = np.random.default_rng(0)\n"
             "    acc = rng.integers(0, 40, size=400)\n"
             "    return pd.DataFrame({'account_id': acc, 'x1': rng.normal(size=400), 'x2': rng.normal(size=400)})\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_disjoint_accounts(self):\n"
             "        train, test, _ = prepare(frame())\n"
             "        self.assertEqual(len(set(train.account_id) & set(test.account_id)), 0)\n"
             "    def test_all_accounts_assigned(self):\n"
             "        df = frame()\n"
             "        train, test, _ = prepare(df)\n"
             "        self.assertEqual(set(train.account_id) | set(test.account_id), set(df.account_id))\n"
             "        self.assertEqual(len(train) + len(test), len(df))\n"
             "    def test_scaler_fit_on_train_only(self):\n"
             "        df = frame()\n"
             "        train, _test, scaler = prepare(df)\n"
             "        self.assertTrue(np.allclose(scaler.mean_, train[['x1', 'x2']].mean().values, atol=1e-6))\n"
             "    def test_reasonable_sizes(self):\n"
             "        train, test, _ = prepare(frame())\n"
             "        self.assertGreater(len(test), 20)\n"
             "        self.assertGreater(len(train), 20)\n",
       hints=["group the split by unique account ids: shuffle accounts, take ~80% of rows worth",
              "a simple deterministic approach: rng.permutation of unique accounts, then map rows by membership",
              "fit the scaler on train[[...]] only"],
       solution="import numpy as np\nimport pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n\n\nFEATURES = ['x1', 'x2']\n\n\ndef prepare(df):\n    rng = np.random.default_rng(0)\n    groups = df[['account_id']].drop_duplicates().reset_index(drop=True)\n    rows_per_group = df.groupby('account_id').size()\n    order = rng.permutation(len(groups))\n    accs = groups['account_id'].to_numpy()[order]\n    target = int(0.8 * len(df))\n    train_acc, total = [], 0\n    for acc in accs:\n        if total >= target:\n            break\n        train_acc.append(acc)\n        total += int(rows_per_group.loc[acc])\n    if not train_acc:\n        train_acc = [accs[0]]\n    train_mask = df['account_id'].isin(set(train_acc))\n    train, test = df[train_mask].copy(), df[~train_mask].copy()\n    scaler = StandardScaler().fit(train[FEATURES])\n    return train, test, scaler\n",
       explanation="Splitting by unique group then assigning whole groups is what GroupShuffleSplit does. Fitting the scaler on train only removes the second leak; the mean_ assertion in the test is exactly the check to write in a code review."),
]


# --------------------------------------------------------------------------- #
#  Projects (§22, §24)
# --------------------------------------------------------------------------- #
def R(dimension, weight=1.0, levels=None) -> dict[str, Any]:
    return {"dimension": dimension, "weight": weight, "levels": levels or {}}


PROJECTS: list[dict[str, Any]] = [
    {
        "slug": "beginner-tabular-analysis", "title": "Titanic-style survival analysis (beginner)",
        "description": "Take a public tabular dataset, clean it, and produce a reproducible analysis plus a baseline model. "
        "Focus: honest data handling, not accuracy. Use your library's chapters on preprocessing/evaluation as your reference.",
        "target_level": "beginner", "domain": "tabular", "difficulty": 1, "est_hours": 12,
        "stack": ["python", "pandas", "matplotlib", "scikit-learn"],
        "skills": ["pd.cleaning", "ml.data_splits", "ml.metrics", "ml.features", "test.unit"],
        "dataset_hint": "Kaggle Titanic, or any open CSV with a target column (UCI breast cancer, penguins).",
        "deliverables": [
            "notebook or scripts: load -> clean -> EDA -> baseline -> evaluated model",
            "a documented feature table (name, type, missing policy, why it is not leakage)",
            "README with how to reproduce and the achieved metric + baseline comparison",
            "one page 'what would break in production'",
        ],
        "milestones": [
            {"title": "Data audit", "description": "Load data, report dtypes/missing/duplicates, define the target and the leakage candidates.",
             "acceptance_criteria": ["a written audit with every column classified", "explicit list of excluded leakage columns"]},
            {"title": "Baseline + split", "description": "Random/majority-class baseline and a stratified split; report both.",
             "acceptance_criteria": ["baseline metric reported alongside model metric", "no preprocessing fit before the split"]},
            {"title": "Pipeline model", "description": "sklearn Pipeline (ColumnTransformer + model) scored with CV.",
             "acceptance_criteria": ["CV mean/std reported", "pipeline object saved (joblib)", "one ablation table"]},
            {"title": "Write-up", "description": "Findings, limits, next experiment.", "acceptance_criteria": ["README complete", "reproduce instructions tested"]},
        ],
        "rubric": [
            R("Data handling", 1.5, {"1": "raw file used directly", "3": "documented cleaning + missing policy", "5": "audit + leakage exclusions justified + validated schema"}),
            R("ML correctness", 1.5, {"1": "no baseline", "3": "baseline + CV", "5": "CV + ablations + calibration/error analysis"}),
            R("Code quality", 1.0, {"1": "one notebook blob", "3": "functions + small tests", "5": "modules, typed, tests, lint"}),
            R("Documentation", 1.0, {"1": "none", "3": "README", "5": "README + decisions log + reproduction verified"}),
            R("Testing", 0.8, {"1": "none", "3": "shape/dtype tests", "5": "golden-metric regression test"}),
        ],
    },
    {
        "slug": "junior-churn-prediction", "title": "Customer churn model with a business metric (junior)",
        "description": "Build a churn predictor where the deliverable is a decision, not a score: cost-aware threshold, "
        "calibration and a retention-impact estimate. Includes a monitored batch scoring job.",
        "target_level": "junior", "domain": "tabular", "difficulty": 2, "est_hours": 25,
        "stack": ["python", "pandas", "scikit-learn", "lightgbm-or-xgboost", "sqlite"],
        "skills": ["ml.data_splits", "adv.imbalance", "ml.calibration", "adv.cost_sensitive", "mle.contracts", "mlops.monitoring_ops"],
        "dataset_hint": "Telco Customer Churn (public), or synthesize with sklearn.datasets.make_classification plus a time column.",
        "deliverables": [
            "train/validate/test protocol with point-in-time splits",
            "calibrated model + threshold chosen under a stated cost matrix",
            "batch scorer writing predictions to sqlite with a run manifest",
            "evaluation report: PR curve, calibration plot, expected saved customers",
        ],
        "milestones": [
            {"title": "Protocol", "description": "Define observation/performance windows, splits, metric and cost matrix before modelling.",
             "acceptance_criteria": ["written protocol doc", "leakage checklist per feature"]},
            {"title": "Model", "description": "Two candidates minimum (linear baseline + boosted trees), CV comparison.",
             "acceptance_criteria": ["CV mean/std for both", "hyperparameters logged", "no test-set peeking"]},
            {"title": "Decision layer", "description": "Calibration + threshold optimisation under the cost matrix.",
             "acceptance_criteria": ["reliability plot", "threshold chosen from a documented trade-off table"]},
            {"title": "Batch job", "description": "Idempotent scoring script with schema validation and logging.",
             "acceptance_criteria": ["re-run overwrites nothing silently", "fails loudly on schema change", "logs metrics"]},
        ],
        "rubric": [
            R("ML correctness", 1.5, {"1": "accuracy only", "3": "PR + calibration", "5": "cost-aware decisioning with sensitivity analysis"}),
            R("Data handling", 1.3, {"1": "random split on rows", "3": "point-in-time split", "5": "split + leakage review + drift check between windows"}),
            R("Code quality", 1.0, {"1": "script", "3": "package + config", "5": "package, typed, tested, documented"}),
            R("Testing", 1.0, {"1": "none", "3": "unit tests for transforms", "5": "unit + golden-metric + schema tests"}),
            R("Monitoring", 1.0, {"1": "none", "3": "logs + counts", "5": "score drift + null-rate alerts + run manifest"}),
            R("Documentation", 0.8, {"1": "none", "3": "README", "5": "README + protocol + decisions + how to operate"}),
        ],
    },
    {
        "slug": "middle-rag-service", "title": "RAG service over a document set (middle)",
        "description": "Build a retrieval-augmented question answering service over 3-10 documents with citations, "
        "an eval harness, and refusal on unsupported questions. The core exercise is grounding, not prompt decoration.",
        "target_level": "middle", "domain": "llm", "difficulty": 3, "est_hours": 35,
        "stack": ["python", "fastapi", "sqlite-or-postgres", "numpy", "optional-llm-api"],
        "skills": ["rag.chunking", "rag.retrieval", "rag.grounding", "mle.api", "llm.evals", "mle.contracts"],
        "dataset_hint": "Any public PDF corpus: arXiv papers you have rights to read, official docs, or your own notes.",
        "deliverables": [
            "ingestion pipeline with structure-aware chunking + metadata (chapter/page)",
            "hybrid retrieval (BM25 + vectors) with an ablation table",
            "FastAPI endpoint returning answer + citations, refusing when unsupported",
            "evaluation: retrieval recall@k, faithfulness on a hand-labelled set of >= 40 questions",
            "README with cost/latency numbers and the failure cases",
        ],
        "milestones": [
            {"title": "Ingestion", "description": "Parsers -> chunks with metadata; inspectable via an endpoint or CLI.",
             "acceptance_criteria": ["chunk viewer", "duplicate/short-chunk filtering", "page/chapter preserved when available"]},
            {"title": "Retrieval", "description": "BM25 + vector + fusion; recall@k on your labelled set.",
             "acceptance_criteria": ["recall@5 table for lexical / vector / hybrid", "documented fusion weights"]},
            {"title": "Generation + grounding", "description": "Answer with numbered citations; refuse when context is insufficient.",
             "acceptance_criteria": ["citation validation against retrieved ids", "refusal path tested", "no invented sources"]},
            {"title": "Service", "description": "API, caching, latency/cost logging, tests.",
             "acceptance_criteria": ["openapi docs", "cache hits measured", "p50/p99 latency logged"]},
            {"title": "Eval report", "description": "Failure taxonomy of at least 10 bad answers.",
             "acceptance_criteria": ["each failure classified: retrieval / generation / prompt / data"]},
        ],
        "rubric": [
            R("Architecture", 1.3, {"1": "one script", "3": "clean modules + boundaries", "5": "swappable providers, config, observability"}),
            R("ML correctness", 1.5, {"1": "no eval", "3": "recall + faithfulness measured", "5": "ablations + error taxonomy + regression suite"}),
            R("Data handling", 1.0, {"1": "fixed chunk size", "3": "structure-aware + metadata", "5": "+ dedupe, incremental reindex, provenance"}),
            R("Testing", 1.2, {"1": "none", "3": "unit + API tests", "5": "eval harness as CI gate with golden answers"}),
            R("Deployment", 1.0, {"1": "none", "3": "Dockerfile", "5": "Docker + healthcheck + rollback notes + load test"}),
            R("Monitoring", 1.0, {"1": "none", "3": "request/latency logs", "5": "cost, citation-miss rate, refusal rate, drift"}),
            R("Code Quality", 1.0, {"1": "mixed", "3": "typed + tested", "5": "typed, tested, reviewed, linted"}),
            R("Monitoring", 1.0, {"1": "none", "3": "request/latency logs", "5": "cost, citation-miss rate, refusal rate, drift"}),
            R("Cost thinking", 0.9, {"1": "ignored", "3": "measured per request", "5": "caching/routing with before/after numbers"}),
        ],
    },
    {
        "slug": "strong-middle-production-api", "title": "Production ML API with CI/CD (strong middle)",
        "description": "Train a model, serve it behind a validated API, and build the release machinery: image, tests as gates, "
        "canary + rollback, dashboards. The model can be simple; the engineering must be serious.",
        "target_level": "strong_middle", "domain": "mlops", "difficulty": 4, "est_hours": 45,
        "stack": ["python", "fastapi", "docker", "github-actions", "prometheus-client", "pytest"],
        "skills": ["mle.api", "mle.serving_perf", "mle.contracts", "mlops.cicd", "mlops.release", "mlops.monitoring_ops", "mle.docker"],
        "dataset_hint": "Reuse the churn model from the junior project, or any tabular model you already have.",
        "deliverables": [
            "versioned model artefact with a manifest (data version, params, metrics)",
            "FastAPI service: /predict, /healthz, /metrics with schema validation and warmup",
            "Docker image + compose; CI running tests, lint, image build, model eval gate",
            "shadow/canary simulation script + rollback runbook",
            "dashboard JSON (or Prometheus queries) for latency, errors, drift",
        ],
        "milestones": [
            {"title": "Artefact discipline", "description": "Model + metadata packaged, loadable by version.",
             "acceptance_criteria": ["manifest with git sha, data version, metrics", "loader unit test"]},
            {"title": "Service", "description": "Inference API with validation, timeouts, batching decision documented.",
             "acceptance_criteria": ["p99 < target measured locally", "422 on bad payloads", "warmup on start"]},
            {"title": "CI/CD", "description": "Tests + eval gate block bad merges; image built on main.",
             "acceptance_criteria": ["failing golden-metric test blocks the pipeline", "reproducible image build"]},
            {"title": "Release engineering", "description": "Shadow scoring, canary with auto-rollback rule.",
             "acceptance_criteria": ["shadow diff report", "documented rollback trigger and rehearsed steps"]},
            {"title": "Observability", "description": "Metrics + alerts + dashboard.",
             "acceptance_criteria": ["drift + null-rate + latency panels", "one alert rule with an owner"]},
        ],
        "rubric": [
            R("Architecture", 1.2, {"1": "monolith", "3": "clear layering", "5": "documented boundaries, seams, failure isolation"}),
            R("Deployment", 1.4, {"1": "manual", "3": "Docker + CI", "5": "CI gates + canary + rollback rehearsed"}),
            R("Monitoring", 1.3, {"1": "none", "3": "latency/errors", "5": "SLIs + drift + label-delay + alert quality reviewed"}),
            R("Scalability", 1.1, {"1": "untested", "3": "load test with results", "5": "capacity model + bottleneck fixed + retested"}),
            R("Code quality", 1.0, {"1": "mixed", "3": "typed + tested", "5": "typed, tested, reviewed, documented, linted"}),
            R("Documentation", 1.0, {"1": "none", "3": "README", "5": "runbook + ADR + onboarding guide"}),
        ],
    },
    {
        "slug": "senior-realtime-recsys", "title": "Real-time recommender / ranking platform (senior)",
        "description": "Design and partially implement a recommendation system with the full production shape: candidate "
        "generation, ranking, freshness, degradation ladder, capacity/cost analysis, offline+online evaluation, and an "
        "architecture doc written as if it would be reviewed by three other teams. Implementation scope: the offline pipeline "
        "plus a serving prototype; the rest must be a defensible design.",
        "target_level": "senior", "domain": "design", "difficulty": 5, "est_hours": 80,
        "stack": ["python", "fastapi", "faiss-or-numpy-ann", "kafka-or-queue-stub", "sql", "docker"],
        "skills": ["adv.recsys_ranking", "design.requirements", "design.data_flow", "design.tradeoffs", "design.recall_rank",
                    "design.online_ml", "design.failure", "design.cost_model", "mlops.monitoring_ops", "senior.ownership"],
        "dataset_hint": "MovieLens-1M / public e-commerce interaction logs; 1M rows is enough to make the sizing real.",
        "deliverables": [
            "RFC: requirements (traffic, p99, freshness, cost cap), 2 architecture options, chosen one, rejected trade-offs",
            "offline pipeline: user/item embeddings + candidate generation evaluated with recall@K and nDCG",
            "serving prototype: ANN retrieval + light ranker + cache, with measured p50/p99 and capacity math",
            "failure design: degradation ladder, timeouts, load shedding, tested with a fault injection script",
            "evaluation plan: offline metrics, A/B design with MDE and guardrails, monitoring/alert definitions",
            "cost model: cost per 1k requests at three traffic levels, with the two biggest levers quantified",
        ],
        "milestones": [
            {"title": "RFC", "description": "Requirements -> options -> decision with numbers.",
             "acceptance_criteria": ["explicit QPS/p99/freshness/cost numbers", "2 alternatives rejected with reasons", "reviewed by an imagined peer"]},
            {"title": "Offline core", "description": "Embeddings + ANN + ranking with honest metrics.",
             "acceptance_criteria": ["recall@50 and nDCG@10 reported", "leakage-safe temporal split", "index build time measured"]},
            {"title": "Serving prototype", "description": "Low-latency path with cache and batching.",
             "acceptance_criteria": ["p99 measured under load", "cache hit ratio reported", "capacity math vs observed"]},
            {"title": "Failure & degradation", "description": "Fault injection + ladder.",
             "acceptance_criteria": ["documented 4-level degradation", "each level triggered in a test", "no hard dependency on one service"]},
            {"title": "Experiment + monitoring", "description": "A/B design + dashboards + alerts.",
             "acceptance_criteria": ["MDE and sample-size calc", "alert with threshold justification", "runbook for the top 3 incidents"]},
            {"title": "Cost review", "description": "Cost per request and optimisation plan.",
             "acceptance_criteria": ["three traffic scenarios costed", "two levers quantified with quality impact"]},
        ],
        "rubric": [
            R("Architecture", 1.5, {"1": "components listed", "3": "justified choices with data flow", "5": "explicit trade-offs, failure analysis, alternatives rejected with numbers"}),
            R("ML correctness", 1.3, {"1": "accuracy only", "3": "ranking metrics with leakage-safe splits", "5": "offline/online metric bridge + experiment validity"}),
            R("Scalability", 1.4, {"1": "none", "3": "capacity math", "5": "sharding/backpressure plan validated by a load test"}),
            R("Reliability", 1.3, {"1": "none", "3": "timeouts + fallback", "5": "degradation ladder + fault-injection tests + SLO/error budget"}),
            R("Cost", 1.1, {"1": "ignored", "3": "rough estimate", "5": "modelled per-request cost with quantified optimisations"}),
            R("Data handling", 1.0, {"1": "single table", "3": "point-in-time features", "5": "contracts + freshness SLAs + quality checks"}),
            R("Testing", 1.0, {"1": "none", "3": "unit + integration", "5": "plus golden-model, load, and chaos tests in CI"}),
            R("Code Quality", 1.0, {"1": "mixed", "3": "typed and tested", "5": "typed, tested, reviewed, documented, linted"}),
            R("Deployment", 1.0, {"1": "manual", "3": "containerised with CI", "5": "canary + rollback rehearsed"}, ),
            R("Monitoring", 1.2, {"1": "none", "3": "latency/error dashboards", "5": "SLOs, drift, label delay, alert quality reviewed"}),
            R("Documentation", 1.2, {"1": "none", "3": "README + design doc", "5": "RFC + ADRs + runbook, reviewable by another team"}),
        ],
    },
]


# --------------------------------------------------------------------------- #
#  Exams (§28)
# --------------------------------------------------------------------------- #
EXAMS: list[dict[str, Any]] = [
    {
        "code": "junior_ml_engineer", "title": "Junior ML Engineer Exam", "target_level": "junior", "min_level": 1,
        "duration_minutes": 75, "passing_score": 68.0,
        "description": "Foundations: Python and data handling, maths you actually use, classical ML correctness, and honest evaluation.",
        "question_codes": ["seed.0.traceback", "seed.1.axis", "seed.1.pandas-dtype", "seed.1.sql-join", "seed.2.dot",
                            "seed.2.derivative", "seed.2.prob-basics", "seed.2.gradient-meaning", "seed.3.leakage",
                            "seed.3.bias-variance", "seed.3.logistic", "seed.3.metrics", "seed.3.cv", "seed.3.trees",
                            "seed.3.sklearn-pipeline", "seed.2.xent"],
        "sections": [
            {"name": "Programming & data", "weight": 20.0, "types": ["mcq", "conceptual", "coding"], "levels": [0, 1]},
            {"name": "Mathematics for ML", "weight": 20.0, "types": ["math", "conceptual"], "levels": [1, 2, 3]},
            {"name": "Classical ML correctness", "weight": 35.0, "types": ["mcq", "open"], "levels": [2, 3, 4]},
            {"name": "Written reasoning", "weight": 25.0, "types": ["open"], "levels": [2, 3], "count": 2},
        ],
    },
    {
        "code": "middle_ml_engineer", "title": "Middle ML Engineer Exam", "target_level": "middle", "min_level": 3,
        "duration_minutes": 105, "passing_score": 70.0,
        "description": "Deep evaluation discipline, DL/LLM mechanics, engineering judgement, and one design answer.",
        "question_codes": ["seed.3.imbalanced", "seed.4.calibration", "seed.4.optuna", "seed.5.vanishing",
                            "seed.5.bn", "seed.5.losscurve", "seed.5.autograd", "seed.6.attention", "seed.6.bpe",
                            "seed.6.chunking", "seed.7.serving", "seed.7.contracts", "seed.8.monitoring",
                            "seed.8.retraining", "seed.9.cost"],
        "sections": [
            {"name": "Advanced ML & DL", "weight": 30.0, "types": ["mcq", "open"], "levels": [4, 5]},
            {"name": "LLM / retrieval", "weight": 20.0, "types": ["open", "mcq"], "levels": [6]},
            {"name": "Engineering & MLOps", "weight": 30.0, "types": ["mcq", "open"], "levels": [7, 8]},
            {"name": "Design under constraints", "weight": 20.0, "types": ["architecture", "open"], "levels": [8, 9]},
        ],
    },
    {
        "code": "senior_ml_engineer", "title": "Senior ML Engineer Assessment", "target_level": "senior", "min_level": 7,
        "duration_minutes": 150, "passing_score": 74.0,
        "description": "System design, trade-offs, scalability, monitoring, cost and production incidents. "
        "Graded on reasoning quality: every claim must be backed by a number, a mechanism, or an explicit assumption.",
        "question_codes": ["seed.9.system-design", "seed.9.cost", "seed.9.incident", "seed.10.leadership",
                            "seed.8.monitoring", "seed.7.repro", "seed.6.rag-eval", "seed.6.kvcache"],
        "sections": [
            {"name": "ML system design", "weight": 30.0, "types": ["architecture"], "levels": [9, 10], "count": 1},
            {"name": "Reliability & incidents", "weight": 20.0, "types": ["open"], "levels": [8, 9, 10]},
            {"name": "Cost & scalability", "weight": 20.0, "types": ["open", "architecture"], "levels": [9]},
            {"name": "Monitoring & evaluation strategy", "weight": 15.0, "types": ["open"], "levels": [8, 9]},
            {"name": "Leadership & judgement", "weight": 15.0, "types": ["open"], "levels": [10]},
        ],
    },
]
