# theory/

Formal results for this fork's research lines. These documents stand on their
own as mathematics: each states its assumptions, proves its claims, and marks
explicitly which claims the experiments did and did not test.

They are deliberately separate from `experiments/` (run-books: how to execute)
and from `exp/<name>/analysis/` (what a particular run measured).

| File | Description |
|------|-------------|
| [markov_inheritance.md](markov_inheritance.md) | Abstract information, risk, selection, and operator results for history-aware retrieval. §8 states boundaries; §9 summarizes the theory. |
| [history_verdict.md](history_verdict.md) | Instantiates the theory for the current pooling → zscore-tanh → non-negative fusion pipeline, proves Propositions 7–11, and separates candidate-loss guarantees from the observed closed-loop SR verdict. |
