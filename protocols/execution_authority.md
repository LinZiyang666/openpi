# Execution Authority — Workflow Regulation

> This document is subordinate legislation under `WORKING_AGREEMENT.md` §9.
> An agent operating under Execution Authority MUST read this document together with the Working Agreement, and MUST NOT read `review_authority.md`. Keep context focused on the authority in effect.
> Any conflict with the Working Agreement is resolved in favor of the Agreement.
> **Entry conditions are defined in Working Agreement §9.2 and are not repeated here.** If you have not yet determined your authority, return to the Working Agreement.

---

## 1. Scope

Execution Authority proceeds strictly stage by stage: plan → G1 → code → G2 → verify → commit → push → documentation. G1 (§3) and G2 (§5) are review gates per Working Agreement §2.4 / §2.6; both invoke the shared §10 Review Process. The agent MUST render a fresh status card at every stage transition.

**Sealed reviewer space**: the directory `tests/review_tests/` is reserved for G2 reviewers' independent probes. Execution Authority — the agent and every sub-agent it spawns — MUST NOT read, list, search, or otherwise inspect anything under that path at any stage. The asymmetry is constitutional: independent tests lose value the moment the audited party can study them.

## 2. Plan

Applies to L2+.

**Required actions**:

- Produce a standalone plan file in `logs/` with the `.log.md` suffix (the rest of the filename is unconstrained), covering: files touched, interfaces introduced or modified, integration points, test strategy, risk register.
- The plan file MUST end with a `## Review Log` section, initially empty. Log conventions and lifecycle are defined in §10.
- Obtain user confirmation of requirements before finalizing the plan (Working Agreement §2.3).

**Prohibitions**:

- Beginning §4 Code before §3 G1 returns `APPROVED`.
- Keeping the plan only in conversation; it MUST be a file on disk.

**Deliverable**:

- Plan file path + concise context summary usable by a G1 reviewer with no conversation history.

**Exit condition**:

- Plan file complete and user-confirmed; proceed to §3 G1.

## 3. G1 — Plan Review Gate

Applies to L2+. Per Working Agreement §2.4.

**Required actions on reach**:

1. State to the user, verbatim: `G1 gate reached. Please initiate a separate Review Authority session to audit <plan-file>.`
2. Halt modifications to the code scope covered by the plan.

**Prohibitions**:

- Satisfying G1 by self-review or by a sub-agent (Working Agreement §9.3).
- Advancing to §4 Code without an `APPROVED` G1 verdict.

**On non-APPROVED verdict**: see §10 Review Process.

**Exit condition**:

- `APPROVED` G1 verdict received; proceed to §4 Code.

## 4. Code

Applies to L1+. For L2+, entered only after `APPROVED` G1 verdict.

**Required actions**:

- Implement the approved plan using interceptor / wrapper / hook patterns (Working Agreement §2.5).
- Preserve backward compatibility upstream, forward compatibility for extensions.
- Minimal changes — no drive-by cleanups, no dead code, no unrelated features (Working Agreement §3.1).
- New features MUST ship with corresponding tests (Working Agreement §6).

**Comment and docstring rules** (Working Agreement §3.2):

- Code comments MUST be in English. No Chinese. No exceptions.
- Every new file MUST have a file-level docstring describing the module's responsibility, public interface, and key dependencies.
- Public classes and public functions MUST have docstrings. Internal helpers (`_` prefix) may omit docstrings unless the logic is non-obvious. Style is descriptive prose; no strict Google/numpy format required.
- Use horizontal-rule comments to delineate logical sections within files; separator width 50–70 `-` characters, visually consistent:

  ```python
  # ------------------------------------------------------------------
  # Section Name
  # ------------------------------------------------------------------
  ```

- Inline comments prefer **why** (rationale) over **what**. Tensor shape annotations are encouraged at key operations (e.g. `# [B, D, H, W]`).
- `TODO` / `FIXME` format: `# TODO(context): description` or `# FIXME(context): description`. `context` is a development phase (e.g. `Step N`), feature name, or person name. Bare `# TODO` / `# FIXME` without context is FORBIDDEN.
- Prohibited patterns: type annotations in comments (use Python native type hints); comments on code the agent did not change; obvious comments that merely restate the code.

**Permitted**:

- Running `uv run pytest` (or any subset) locally as a self-check at the executor's discretion. Such runs have **no procedural force**: they do not satisfy §6 Verify, do not gate commit, and do not authorize skipping §6. Only the §6 Verify run carries procedural weight.

**Prohibitions**:

- Deviating from the approved plan without flagging to the user and securing renewed approval.
- Leaving dead code commented out (Working Agreement §3.1).
- Non-English comments, bare `# TODO` / `# FIXME`, type annotations in comments, comments on code the agent did not change (Working Agreement §3.2).

**Deliverable**:

- Complete diff.
- Plan-conformance statement: "code fully follows the approved plan" or "deviations: <list>, user-consented at <point>".
- Any local test output the executor produced under the Permitted clause, attached as evidence (not a substitute for §6 Verify).

**Exit condition**:

- For L2+: proceed to §5 G2.
- For L1: proceed directly to §6 Verify.

## 5. G2 — Code Review Gate

Applies to L2+. Per Working Agreement §2.6.

**Required actions on reach**:

1. All changes staged but NOT committed.
2. Diff summary, plan-conformance statement, and any §4 local test output (advisory only — procedural test authority lies with §6 Verify) produced.
3. State to the user, verbatim: `G2 gate reached. Please initiate a separate Review Authority session for code audit.`

**Prohibitions**:

- Satisfying G2 by self-review or by a sub-agent (Working Agreement §9.3).
- Advancing to §6 Verify without an `APPROVED` G2 verdict AND completion of the final polish in §10.

**On non-APPROVED verdict**: see §10 Review Process.

**Exit condition**:

- `APPROVED` G2 verdict received AND final polish (§10) performed; proceed to §6 Verify.

## 6. Verify

Applies to L1+.

**Required actions**:

- Run `uv run pytest` after the §10.3 final polish and capture full output (Working Agreement §2.7). This is the run with procedural force; it gates §7 Commit. Earlier §4 runs do not satisfy this requirement.
- For inference-path changes, run the staged API tests (Working Agreement §2.7).
- `@pytest.mark.manual` marks tests requiring GPU or external services. CI (`.github/workflows/test.yml`) runs only non-manual tests; manual tests are run locally by the executor when the change touches the gated paths.

**Prohibitions**:

- Proceeding to §7 Commit with any failing test.
- Silencing, skipping, or marking tests as `xfail` to bypass failures.
- Substituting any §4 local test output for the §6 Verify run.

**Deliverable**:

- Test log + one-line pass / fail summary.

**Exit condition**:

- All required tests pass; summary delivered.

## 7. Commit

**Tooling configuration** (Working Agreement §7):

- Formatting & linting: `pyproject.toml` `[tool.ruff]`
- Pre-commit hooks: `.pre-commit-config.yaml`
- Editor settings: `.vscode/settings.json`

**Required actions**:

- Pre-commit hooks MUST pass (Working Agreement §7).
- Commit messages describe **why**, not **what** (Working Agreement §7).

**Prohibitions**:

- Adding `Co-Authored-By` lines (Working Agreement §7).
- Committing without explicit user instruction.

**Deliverable**:

- Commit SHA and message.

**Exit condition**:

- `git status` confirms the commit succeeded.

## 8. Push

**CI pipeline configuration** (Working Agreement §7): `.github/workflows/` — a push to a branch under CI triggers these workflows automatically.

**Required actions**:

- Perform a push only on explicit user instruction.

**Prohibitions**:

- Pushing without explicit user instruction.
- `git push --force` to `main` or `master` under any circumstances.
- `git push --force` to any branch without explicit, specific user instruction for that push.

**Deliverable**:

- Remote confirmation (push output or PR URL).

**Exit condition**:

- User acknowledges the remote state.

## 9. Documentation

**Location map** (Working Agreement §4):

| Location | Purpose |
|----------|---------|
| `docs/` | Project architecture docs and usage guides |
| `logs/` | Implementation plans, design decisions, code change records |
| `weekly_plan.md` | Current week's working plan |
| `exp/` | Experiment scripts and configs — NO design docs |

**When to create**:

- New subsystem → architecture doc in `docs/`.
- Non-obvious design decision → log in `logs/`.
- Minor change derivable from code → do NOT create a document.

**Documentation language**: defaults to Chinese. English translations of Chinese logs (`foo.log.md` → `foo.en.log.md`) are recommended but not mandatory. Code comments are always English (§4).

**Log status system** (Working Agreement §5):

| Status | Location | Meaning |
|--------|----------|---------|
| `In Progress` | `logs/` top level | Actively being executed or updated |
| `Plan` | `logs/` top level | Task breakdown, not yet implemented |
| `Design Only` | `logs/` top level | Design exists, implementation not confirmed |
| `Implemented` | `logs/archive/` | Code exists, validation pending |
| `Validated` | `logs/archive/` | Implemented and verified |
| `Done-High-Risk` | `logs/archive/` | Code landed, no test coverage |
| `Historical` | `logs/archive/` | Historical reference, no longer active source of truth |

**Required actions**:

- `docs/` changes → `docs/README.md` updated in the same commit (Working Agreement §4).
- `logs/` changes → `logs/README.md` updated in the same commit (Working Agreement §4).
- Completed logs move to `logs/archive/`; confirm final log status with the user before archiving (Working Agreement §5).

**Prohibitions**:

- Creating design documents inside `exp/` (Working Agreement §4).
- Deleting any file under `logs/archive/` (Working Agreement §5).

**Deliverable**:

- Index diff showing the new / moved / removed entry, in the same commit as the doc change.

**Exit condition**:

- Doc files and corresponding index are consistent.

## 10. Review Process

Shared protocol invoked by §3 G1 and §5 G2.

### 10.1 Review Log conventions

- The plan file's `## Review Log` section is the sole channel for iterative review discussion and the sole part of the plan file the reviewer may write.
- The Log is append-only until the final polish (§10.3). Deleting, rewriting, or reordering existing entries — whether reviewer or executor — is a violation.
- Entries are grouped under round headers of the form `### <G1|G2> Round N — <Reviewer|Executor>`, with N monotonically increasing.

### 10.2 Iteration (on non-APPROVED verdict)

1. Read every reviewer entry appended since the last handoff.
2. For each item (question, concern, suggestion, reasoning), evaluate with project-specific expertise. The executor is the authoritative voice on project substance; the reviewer challenges and probes, not dictates.
3. If well-founded: modify the plan body (G1) or the code (G2) accordingly, and append `- Accepted — <summary of change>` under a new `### <G1|G2> Round N — Executor` header.
4. If not well-founded: append `- Rejected — <explicit reasoning grounded in project facts>`. Silent or unreasoned rejection is a violation.
5. Every reviewer item MUST receive exactly one executor response; skipping items is a violation.
6. Re-enter the corresponding gate (§3 or §5) with the updated artifact and the appended response round.

### 10.3 Final polish (after `APPROVED` G2)

Before leaving §5 G2 and entering §6 Verify:

1. Polish the plan body: resolve remaining TBDs, finalize language, ensure the plan reads as a clean, coherent record of what was built.
2. Delete the entire `## Review Log` section — every reviewer entry, every executor response, every round header across G1 and G2.
3. Stage the polished plan file as part of the change set.

## 11. Violation Consequences

Any action that properly belongs to Review Authority (self-review, self-approval, rubber-stamping a sub-agent's audit) is void and MUST be re-reviewed by a new, independent Review Authority session.

Stage violations (e.g., skipping §2 Plan or §3 G1 on an L2+ task, deviating from the approved plan without renewed approval, rejecting a reviewer item without reasoning, pushing without confirmation) likewise invalidate the affected work; the agent MUST return to the earliest violated stage and proceed from there.
