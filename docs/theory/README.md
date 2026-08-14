# theory/

Formal results for this fork's research lines. These documents stand on their
own as mathematics: each states its assumptions, proves its claims, and marks
explicitly which claims the experiments did and did not test.

They are deliberately separate from `experiments/` (run-books: how to execute)
and from `exp/<name>/analysis/` (what a particular run measured).

| File | Description |
|------|-------------|
| [markov_inheritance.md](markov_inheritance.md) | **Markov inheritance law** for training-free retrieval caches. Lemma 1 (label conditional independence under a memoryless teacher), **Lemma 2 (denoising bound: `I(a*;h|k) ≤ I(a*;o|k)`)**, Proposition 3 (phase absorption — the formal statement E1-O tests), Theorem 4 + Corollary 4.1 (imitation-error improvement equals `I(a*;h|k)`, hence bounded by the key's lossiness), Proposition 5 (success filtering is collider conditioning — the one channel that re-injects history, with a closed-form two-branch example), Proposition 6 (the additive non-negative scoring class cannot express a difference kernel). §8 maps each claim to the experiment that tested it; §9 lists what the document does **not** claim. Companion experiment: [`exp/markov_sufficiency/`](../../exp/markov_sufficiency/analysis/synthesis.md) |
