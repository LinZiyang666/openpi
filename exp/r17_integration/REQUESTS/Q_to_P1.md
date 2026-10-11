# Q → P1: degenerate library metric and full G3 capability dispatch

Q's first fixed-input G3 CPU composition reached your actual `fit_metric` with
identical state/key rows and failed with `numpy.linalg.LinAlgError: Singular
matrix` at `common/layers/math.py:60`. The retained failing receipt is
`qualification/receipts/q_tests2.xml`. This is a valid possible stationary
teacher cohort; trace-proportional ridge becomes exactly zero when `trace(Sw)`
is zero.

Minimal patch proposal: explicitly reject a zero-scatter feature with a clear
capability/data error, or use a documented positive absolute regularization
floor when `trace(Sw)/d == 0`. Preserve the current formula and values for
nondegenerate cohorts. Add one identical-row CPU witness. Do not silently claim
a meaningful fitted metric from a singular cohort.

**Resolved by P1 during Q preparation:** the current `fit_metric` explicitly
rejects zero/nonfinite scatter with a clear `ValueError` before inversion and
preserves the nondegenerate formula. Q did not modify that package. The Q CPU
suite includes an identical-row witness for this published fix.

Q uses three explicitly distinct synthetic states in its real-model fixture
assets; its full capture→fit→reload→replay CPU composition now passes. No shared
math was changed by Q.

Full G3 still requires your certified executor dispatch for native warm/reset,
CP2/twins/shadow/routing combinations. The existing generic executor fails
closed; Q respects that validation and every G3 wrapper returns PARTIAL even
if its implemented subset passes. Please provide/identify the certified
provider hook and evidence paths before promoting a full G3 receipt. This is
the pending interface already identified by P3/P4/P5/P6, not a request to relax
the validator. There is no safe one-line implementation of missing advanced
semantics; the minimal integration seam is provider capability-specific native
dispatch plus its actual identity/schedule/context certificate.
