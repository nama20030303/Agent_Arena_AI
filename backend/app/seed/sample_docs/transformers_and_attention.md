# Transformers: Attention, Position and Why They Replace RNNs

*ML Engineer Academy sample material — self-authored. Chapter 7.*

## 7.1 Self-attention as soft, learned retrieval

Given a sequence of token representations `X ∈ ℝ^{n×d}`, three projections produce queries, keys and
values. The output for position *i* is a weighted average of all values, where the weights are
normalised similarities between its query and every key:

```text
Q = XW_Q,  K = XW_K,  V = XW_V
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
```

Three facts to be able to state from memory:

1. **Path length.** Any two positions are one matmul apart, so gradients do not have to traverse a
   chain — the reason transformers train more easily than RNNs on long sequences.
2. **Cost.** Attention is `O(n²·d)` in compute and `O(n²)` in memory. Everything interesting about
   long-context engineering is a response to that quadratic term.
3. **Permutation equivariance.** Without a position signal, the attention output is the same for any
   reordering of the input — so position must be injected explicitly.

The `1/√d_k` factor is variance control: if the components of q and k are roughly unit-variance and
independent, `q·k` has variance ≈ d_k. Unscaled, the softmax saturates, its Jacobian collapses, and
gradients vanish.

## 7.2 Multi-head attention

Split the model dimension into `h` heads of size `d/h`, attend in each subspace, concatenate, project:

```text
head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)
MultiHead = Concat(head_1..head_h) W_O
```

Heads let the model attend to different relations at once (syntactic, positional, coreferential,
induction patterns). Total parameter count is roughly the same as single-head at full width — the
win is expressiveness, not capacity. Cost note: with `h` heads of size `d/h`, the `n²` score matrix
cost is unchanged in aggregate.

## 7.3 Positional information

- **Sinusoidal (original)**: fixed functions of position; extrapolates poorly in practice.
- **Learned embeddings**: add a `n_max × d` table; no relative notion, fails past `n_max`.
- **RoPE (rotary)**: rotate q/k by an angle proportional to position, so the dot product depends on
  *relative* offset. Modern LLMs use it, and length-extension tricks (e.g. NTK-aware scaling,
  position interpolation) work by rescaling the rotation frequencies.
- **ALiBi**: add a per-head linear distance penalty to the scores — an explicit recency prior, which
  is why it extrapolates to longer contexts gracefully.

## 7.4 The block, and why it trains

```text
h = x + Attention(LayerNorm(x))          # residual + pre-norm
y = h + FFN(LayerNorm(h))                # FFN = Linear(d→4d) → GELU → Linear(4d→d)
```

- The residual stream gives a shallow gradient path; layer norm stabilises scale; pre-norm trains
  more stably at depth than the original post-norm.
- The FFN holds most of the parameters and behaves like a key-value memory (the "knowledge neurons"
  intuition): attention moves information between positions, the FFN transforms it per position.
- Stacking L blocks gives L refinement steps of the same representation — which is why deleting or
  bypassing blocks (early exit, layer pruning) is often survivable.

## 7.5 Causal masking, KV cache and decoding

In a language model, position *i* may attend only to positions ≤ i: implement it by adding `-inf` to
the upper triangle of the score matrix before the softmax. At inference, generation is autoregressive:

- **prefill**: process the whole prompt in parallel — compute-bound, uses the GPU well.
- **decode**: one token at a time — memory-bandwidth-bound; you re-read weights per token.
- **KV cache**: keys/values of past tokens never change, so cache them; each step computes QKV only
  for the new token. Cost: `2 · L · n · d · bytes` per sequence, per layer pair of (K,V). That buffer
  is why long contexts are memory-bound and why techniques like multi-query attention (shared K/V
  heads), grouped-query attention, paged KV memory and KV quantisation exist.
- Sampling: greedy, temperature, top-k, top-p (nucleus), min-p; beam search helps transcription-like
  tasks and hurts open-ended generation.

## 7.6 Encoder vs decoder vs encoder-decoder

| Family | Attention | Typical use |
|---|---|---|
| Encoder (BERT-like) | bidirectional | classification, embedding/retrieval |
| Decoder (GPT-like) | causal | generation |
| Encoder-decoder (T5-like) | bidirectional encoder + cross-attention decoder | seq2seq, translation, summarisation |

For retrieval embeddings you want an *encoder*; for instruction following you want a *causal decoder*.
Mixing them up is a common architecture mistake in RAG systems — a bidirectional model trained for
cosine similarity is a poor generator, and vice versa.

## 7.7 Fine-tuning and its budgets

Full fine-tuning costs optimizer state (≈2 bytes/param for Adam moments at bf16 plus gradients), which
is why parameter-efficient methods exist. LoRA freezes `W` and trains `ΔW = BA` with `B ∈ ℝ^{d×r}`,
`A ∈ ℝ^{r×d}`, `r ≪ d`; at inference `BA` can be merged, so latency is unchanged. QLoRA adds a
quantised (4-bit) frozen base plus paged optimisers, making single-GPU fine-tuning feasible.
For alignment after supervised fine-tuning: DPO/RLHF-style preference optimisation changes *what the
model prefers*, not what it knows — so its data quality requirements (paired preferences, matched
format) are different and stricter.

## 7.8 Failure modes to be able to name

- Attention entropy collapse (one hot) after long training → loss spikes; mitigated by logit soft-capping.
- Positional out-of-distribution: performance degrades beyond trained length despite "supporting" more.
- Repetition degeneration from temperature < 1 plus long contexts; fixed by sampling and penalty terms.
- Cross-entropy on tokens weights every token equally — retrieval-quality evaluation needs a
  separate metric (this is why an LLM app needs its own eval harness).

## 7.9 Exercises

1. Show that `q·k` has variance ≈ d_k when components are unit-variance and independent, and derive the √d_k factor.
2. Compute the KV-cache size for a 32-layer, 4096-dim, 32-head model at 8k context, batch 16, fp16.
3. Implement causal self-attention for `h` heads in NumPy for one batch and verify it equals the
   masked full-attention computation for the last position.
4. Explain why pre-norm is more robust to depth than post-norm, in terms of the gradient path.
5. You need semantic search over 20M passages on a budget. Argue for encoder-vs-decoder embeddings,
   and give the index/memory arithmetic for the ANN layer.
