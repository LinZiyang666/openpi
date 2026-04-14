# OpenPI Working Agreement

This document is the **sole authoritative source of rules** for this project. Every human developer and AI agent participating in this project **MUST** comply with this Working Agreement. Ignorance is not an excuse. Violation is not tolerable.

> In case of conflict with any other document (CLAUDE.md, docs/, logs/README.md, etc.), **this file prevails. Always.**

**Project Owner**: Ziyang Lin. Holds absolute authority over this Working Agreement and all project matters. May override any process at will.

---

## AI Agent Directive

Agent behavioral rules and session initialization protocol are embedded in [`CLAUDE.md`](CLAUDE.md) (auto-loaded). Not repeated here.

---

## 1. Scope

This project is a fork of Physical-Intelligence/openpi, targeting **PyTorch + Pi0.5 only**. The JAX path is disabled.

Technical details: [`docs/reference/openpi.md`](docs/reference/openpi.md).

---

## 2. Development Workflow

### 2.1 Work Classification

All work is classified into four levels. Each level dictates the required process. **There are no shortcuts.**

| Level | Definition | Examples | Required Process |
|-------|-----------|----------|-----------------|
| **L0 Trivial** | Cosmetic changes with zero logic impact | Typo, formatting, pure doc/log changes | Direct commit |
| **L1 Minor** | Localized change, bounded impact | Small bug fix, config tweak, `exp/` scripts | Code → Verify |
| **L2 Standard** | Feature modification or new component | New KeyBuilder, refactor SearchStrategy | Understand → Plan → **G1** → Code → **G2** → Verify |
| **L3 Architectural** | Cross-module change or new subsystem | New checkpoint level, interface change | Understand → Plan → **G1** → Code → **G2** → Verify + architecture doc update |

**Level is determined by the executor.** Disputes are resolved by the project owner. **Deliberately downgrading a level to skip gates is a Working Agreement violation.**

### 2.2 Understand Before Coding (L1+)

- Read [`docs/README.md`](docs/README.md) and [`logs/README.md`](logs/README.md) indexes first, then the relevant specific documents.
- **Writing code without understanding the existing architecture and prior decisions is FORBIDDEN.**

### 2.3 Plan Before Implementation (L2+)

1. Review architecture docs and discuss requirements with the user.
2. Produce a code plan: which files, what interfaces, how it integrates with existing modules.
3. Obtain user confirmation before proceeding.

### 2.4 Plan Review Gate — G1 (L2+)

> **BLOCKING GATE**: After the plan is complete, it **MUST** be reviewed by a party other than the plan author (human developer or independent review agent). No plan, no code. No review, no code. No exceptions.

Review checklist:
- Architecture consistency (does it conform to existing design patterns?)
- Interface compatibility (backward/forward compatible?)
- Risk identification (boundary conditions, performance concerns, coupling risks)
- Test strategy

**Release criterion**: Reviewer explicitly states "plan approved". If not approved, the plan MUST be revised until both parties reach agreement.

### 2.5 Code (L1+)

- New features MUST be decoupled from the inference pipeline — use interceptor / wrapper / hook patterns.
- Backward compatible with upstream, forward compatible with extensions.
- Prefer composable wrappers over modifying inference internals.

### 2.6 Code Review Gate — G2 (L2+)

> **BLOCKING GATE**: After coding is complete, the code **MUST** be reviewed by a party other than the author (human developer or independent review agent). Unreviewed code does not ship. Period.

Review checklist:
- Consistency with the approved plan
- Test coverage and all tests passing
- Documentation and indexes updated
- No regressions

**Release criterion**: Reviewer explicitly states "code approved". If not approved, the code MUST be fixed until both parties reach agreement.

Emergency hotfixes may skip G1, but G2 still requires post-hoc review.

### 2.7 Verify (L1+)

- `uv run pytest` — all tests MUST pass.
- Inference path changes MUST pass staged API tests.

---

## 3. Code Standards

### 3.1 Principles

- **Minimal change**: Change only what is required. No drive-by cleanups. No extra features.
- **No dead code**: Unused code is deleted, not commented out.
- **Decoupling**: New features plug in via interceptor / wrapper / hook. Do NOT modify inference internals.

### 3.2 Commenting Rules

**Language**: All code comments MUST be in English. No Chinese. No exceptions.

**Module Docstring**: Every new file MUST have a file-level docstring describing the module's responsibility, public interface, and key dependencies.

**Class / Function Docstring**:
- Public classes and public functions MUST have docstrings.
- Internal helpers (`_` prefix) may omit docstrings unless the logic is non-obvious.
- Style: Multi-line, descriptive prose. No strict Google/numpy format required, but responsibilities must be clearly stated.

**Section Separators**:
- Use horizontal rule comments to delineate logical sections within files:
  ```python
  # ------------------------------------------------------------------
  # Section Name
  # ------------------------------------------------------------------
  ```
- Separator width: 50–70 `-` characters, visually consistent.

**Inline Comments**:
- Prefer **why** (rationale) over **what** (the code already says what).
- Exception: Tensor shape annotations are encouraged at key operations: `# [B, D, H, W]`

**TODO / FIXME**:
- Format: `# TODO(context): description` or `# FIXME(context): description`
- `context` may be a development phase (e.g., `Step N`), feature name, or person name.
- **Bare `# TODO` or `# FIXME` without context is FORBIDDEN.**

**Prohibited**:
- Do NOT write type annotations in comments (use Python native type hints).
- Do NOT add comments to code you did not change.
- Do NOT write obvious comments (e.g., `# increment counter` next to `i += 1`).

### 3.3 Tooling

Code formatting and linting are enforced automatically by the toolchain. Configuration lives in:
- Formatting & Linting: `pyproject.toml` `[tool.ruff]`
- Pre-commit hooks: `.pre-commit-config.yaml`
- Editor: `.vscode/settings.json`

**Developers need only ensure pre-commit hooks pass. Memorizing specific rules is not required.**

---

## 4. Documentation Standards

### 4.1 Where to Write

| Location | Purpose |
|----------|---------|
| `docs/` | Project architecture docs and usage guides |
| `logs/` | Implementation plans, design decisions, code change records |
| `weekly_plan.md` | Current week's working plan |
| `exp/` | Experiment scripts and configs — **NO design docs** |

### 4.2 When to Create

- **New subsystem** → Write an architecture doc in `docs/`
- **Non-obvious design decision** → Write a log in `logs/`
- **Minor change or derivable from code** → Do NOT create a document

### 4.3 Index Sync Rule

**After every doc creation/modification/move, the corresponding index MUST be updated immediately:**
- `docs/` files → `docs/README.md`
- `logs/` files → `logs/README.md`

**Failure to sync indexes is a violation. "I'll update it later" is not acceptable.**

### 4.4 Language

- Documentation defaults to Chinese.
- English translations of Chinese logs are recommended (`foo.log.md` → `foo.en.log.md`), not mandatory.
- Code comments MUST be in English (see §3.2).

---

## 5. Log Management

### 5.1 Status System

| Status | Location | Meaning |
|--------|----------|---------|
| `In Progress` | `logs/` top level | Actively being executed or updated |
| `Plan` | `logs/` top level | Task breakdown, not yet implemented |
| `Design Only` | `logs/` top level | Design exists, implementation not confirmed |
| `Implemented` | `logs/archive/` | Code exists, validation pending |
| `Validated` | `logs/archive/` | Implemented and verified |
| `Done-High-Risk` | `logs/archive/` | Code landed, no test coverage |
| `Historical` | `logs/archive/` | Historical reference, no longer active source of truth |

### 5.2 Lifecycle Rules

1. Active work (`In Progress` / `Plan` / `Design Only`) stays at `logs/` top level.
2. Completed work moves to `logs/archive/`.
3. **NEVER delete archived logs.** They are permanent historical records.
4. Confirm final status with the user before archiving.
5. Sync `logs/README.md` on every file change.

---

## 6. Testing Standards

- New features MUST have corresponding tests.
- `@pytest.mark.manual` marks tests requiring GPU or external services.
- CI automatically runs non-manual tests (see `.github/workflows/test.yml`).

---

## 7. Git & CI Standards

- Main branch: `main`. Dev branches: named by developer or feature.
- Pre-commit hooks MUST pass before every commit.
- Commit messages describe **why**, not **what**.
- Do NOT add Co-Authored-By lines to commit messages.
- CI pipeline configuration: `.github/workflows/`

---

## 8. Subsystem Rules

Subsystem-specific architecture constraints and component rules are defined in their respective `docs/` files. Once approved and registered by the project owner, they carry **the same authority as this Working Agreement**.

| Subsystem | Rule Documents |
|-----------|---------------|
| Cache System | [`docs/architecture/cache_system.md`](docs/architecture/cache_system.md), [`docs/cache/tutorial.md`](docs/cache/tutorial.md) |
| Data Collection | [`docs/data_collection/guide.md`](docs/data_collection/guide.md) |

**New subsystem rule documents MUST be approved by the project owner before registration in this table.**

---

## 9. Amendment Process

- **Proposal**: Any participant may propose amendments to this Working Agreement.
- **Approval**: All amendments MUST be approved by the project owner (Ziyang Lin) before taking effect.
- **Tracking**: Amendment history is tracked via git history. No separate change log is maintained.
