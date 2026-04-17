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

Review checklist and procedure are defined in [`protocols/review_authority.md`](protocols/review_authority.md) §4.

**Release criterion**: Reviewer explicitly states "plan approved". If not approved, the plan MUST be revised until both parties reach agreement.

### 2.5 Code (L1+)

- New features MUST be decoupled from the inference pipeline — use interceptor / wrapper / hook patterns.
- Backward compatible with upstream, forward compatible with extensions.
- Prefer composable wrappers over modifying inference internals.

### 2.6 Code Review Gate — G2 (L2+)

> **BLOCKING GATE**: After coding is complete, the code **MUST** be reviewed by a party other than the author (human developer or independent review agent). Unreviewed code does not ship. Period.

Review checklist and procedure are defined in [`protocols/review_authority.md`](protocols/review_authority.md) §4.

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

### 3.2 Comments

- Code comments MUST be in English. No Chinese. No exceptions.
- Every new file MUST have a module-level docstring; public classes and public functions MUST have docstrings.

Detailed commenting conventions (docstring style, section separators, TODO/FIXME format, prohibited patterns) are defined in [`protocols/execution_authority.md`](protocols/execution_authority.md) §4.

### 3.3 Tooling

Formatting, linting, and pre-commit configuration paths are defined in [`protocols/execution_authority.md`](protocols/execution_authority.md) §7. Developers need only ensure pre-commit hooks pass.

---

## 4. Documentation

**Location scope** (constitutional):

- `docs/` — project architecture docs and usage guides.
- `logs/` — implementation plans, design decisions, code change records.
- `weekly_plan.md` — current week's working plan.
- `exp/` — experiment scripts and configs. **NO design docs.**

**Index Sync Rule (constitutional red line)**: after every doc creation / modification / move in `docs/` or `logs/`, the corresponding README (`docs/README.md` or `logs/README.md`) MUST be updated in the same commit. "I'll update it later" is not acceptable. Failure to sync indexes is a violation.

Location policy detail (when to create, language convention, documentation language default) is defined in [`protocols/execution_authority.md`](protocols/execution_authority.md) §9.

---

## 5. Log Management

**Lifecycle red lines (constitutional)**:

- Active work stays at `logs/` top level; completed work moves to `logs/archive/`.
- **NEVER delete archived logs.** They are permanent historical records.
- Confirm final status with the user before archiving.
- Sync `logs/README.md` on every file change.

The full status taxonomy (status names, locations, meanings) is defined in [`protocols/execution_authority.md`](protocols/execution_authority.md) §9.

---

## 6. Testing Standards

- New features MUST have corresponding tests.
- All non-manual tests MUST pass CI.

Test markers and CI configuration details are defined in [`protocols/execution_authority.md`](protocols/execution_authority.md) §6.

---

## 7. Git & CI Standards

**Constitutional red lines**:

- Main branch: `main`. Dev branches: named by developer or feature.
- Pre-commit hooks MUST pass before every commit.
- Commit messages describe **why**, not **what**.
- Do NOT add Co-Authored-By lines to commit messages.
- `git push --force` to `main` / `master` is forbidden under any circumstance.

Tooling configuration paths (pre-commit, ruff, CI workflow) are defined in [`protocols/execution_authority.md`](protocols/execution_authority.md) §7 and §8.

---

## 8. Subsystem Rules

Subsystem-specific architecture constraints and component rules are defined in their respective `docs/` files. Once approved and registered by the project owner, they carry **the same authority as this Working Agreement**.

| Subsystem | Rule Documents |
|-----------|---------------|
| Cache System | [`docs/architecture/cache_system.md`](docs/architecture/cache_system.md), [`docs/cache/tutorial.md`](docs/cache/tutorial.md) |
| Data Collection | [`docs/data_collection/guide.md`](docs/data_collection/guide.md) |
| Experiment Artifact Layout | [`docs/experiments/artifact_layout.md`](docs/experiments/artifact_layout.md) |

**New subsystem rule documents MUST be approved by the project owner before registration in this table.**

---

## 9. Agent Authority Separation

### 9.1 Two Authorities

All agent activity in this project MUST be conducted under exactly one of the
following two authorities at any given time:

| Authority | Purpose | Regulation (outward pointer) |
|-----------|---------|------------------------------|
| **Execution Authority** | Understand, plan, code, verify, commit, push | [`protocols/execution_authority.md`](protocols/execution_authority.md) |
| **Review Authority** | G1 plan review, G2 code review, independent audit | [`protocols/review_authority.md`](protocols/review_authority.md) |

### 9.2 Entry Conditions

Authority is determined at session start, **before any subordinate regulation is loaded**.

**Execution Authority (default)**: the session enters Execution Authority when either holds:

- A new session starts and the user has not declared Review Authority; OR
- The user explicitly instructs the agent to operate under Execution Authority.

**Review Authority**: the session enters Review Authority if, and only if, the user explicitly instructs the agent to operate under it. Typical forms:

- `conduct G1 review of <plan-file>`
- `conduct G2 review of branch <name>`
- `audit <component> for <concern>`

**Entry obligation**: the session status card MUST declare `Authority: Execution` or `Authority: Review` as part of its first render on entry. Only after that declaration may the agent load the corresponding subordinate regulation.

### 9.3 Mutual Exclusion

A single agent — **including every sub-agent it spawns within its own session** — MUST NOT hold Execution and Review authority simultaneously.

- The initiating session declares its authority at session start and holds that authority for the session's full duration.
- Every sub-agent created by that session inherits the same authority and is bound by the same restrictions.
- G1 and G2 reviews (which by §2.4 and §2.6 require independence) MUST be performed by a separately-initiated Review Authority session. They MUST NOT be performed by the author session or any of its sub-agents.

### 9.4 On-Demand Loading

Each regulation is self-contained. An agent MUST read only the regulation corresponding to its declared authority. Reading the opposite regulation is unnecessary and discouraged to keep context focused.

---

## 10. Amendment Process

- **Proposal**: Any participant may propose amendments to this Working Agreement.
- **Approval**: All amendments MUST be approved by the project owner (Ziyang Lin) before taking effect.
- **Tracking**: Amendment history is tracked via git history. No separate change log is maintained.
