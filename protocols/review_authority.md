# Review Authority — Workflow Regulation

> This document is subordinate legislation under `WORKING_AGREEMENT.md` §9.
> An agent operating under Review Authority MUST read this document together with the Working Agreement, and MUST NOT read `execution_authority.md`. Keep context focused on the authority in effect.
> Any conflict with the Working Agreement is resolved in favor of the Agreement.
> **Entry conditions are defined in Working Agreement §9.2 and are not repeated here.** If you have not yet determined your authority, return to the Working Agreement.

---

## 1. Scope

Review Authority is the sole authority empowered to execute Working Agreement §2.4 (G1 Plan Review Gate) and §2.6 (G2 Code Review Gate), and to perform independent audits assigned by the project owner. The review proceeds strictly stage by stage: intake → context acquisition → assessment → verdict delivery. Each stage is legislated in §2–§5.

## 2. Intake

**Required actions**:

- Confirm review type: `G1`, `G2`, or `Audit`.
- Confirm the specific target: plan file path (G1 and G2 both reference the same plan file), branch / commit range / diff artifact (G2), or audit target.
- Declare `Authority: Review` and the review type in the status card.

**Prohibitions**:

- Proceeding without an unambiguous review type and target. Ambiguous requests MUST be clarified with the user, not guessed.
- Accepting a review of the agent's own prior-session work. Such a request MUST be rejected and returned to the user.

**Deliverable**:

- Intake statement of the form: `Review type: <G1|G2|Audit>; target: <path or identifier>; checklist: <applicable §>.`

**Exit condition**:

- Intake statement delivered and the user has not objected.

## 3. Context Acquisition

**Required actions**:

- Read the target material in full (plan file for G1; diff + changed files for G2; target scope for Audit).
- For G2, read the polished plan file. The G1 Review Log was deleted at the Post-G1 polish (`execution_authority.md` §3.1) and is NOT available — the polished plan body is the sole record of the agreed G1 design. Any prior G2 rounds in `## Review Log` MUST be read in full so the present round builds on, rather than duplicates, prior G2 exchange.
- For iterative rounds (Round N≥2), read the executor's responses appended since the last Review Authority session, and use `git diff` (working tree vs the prior reviewer-staged index) to surface every plan/code change the executor introduced this round. Round 1 has no prior staged baseline; read the working tree as-is. The current round MUST address any executor rejections the reviewer still disagrees with, not silently drop them.
- Read the referenced upstream: the approved plan (for G2), task description, relevant subsystem charters (Working Agreement §8), and `docs/` / `logs/` files referenced by the target.
- Read any §4 advisory test output the executor attached. Note that the procedural §6 Verify run has not yet occurred at G2; the reviewer's "tests passing" judgment rests on this advisory output combined with the reviewer's own §3.1 independent runs.

**Prohibitions**:

- Modifying any file. Two narrow exceptions are carved out elsewhere: appending to the plan file's `## Review Log` section, reserved for §5 Verdict Delivery; and creating / editing files under `tests/review_tests/` during G2, governed by §3.1 below.
- Running state-changing shell actions, including "helper" lint or format fixes. Test execution under §3.1 is exempt.
- Spawning writable sub-agents. Only read-only `Explore` sub-agents are permitted.
- Speculating on missing material. Incomplete material MUST trigger a specific request to the user and a pause.

### 3.1 Independent testing (G2 only)

The G2 reviewer holds standing authority to design and run independent tests as part of context acquisition:

- File writes are permitted only under `tests/review_tests/`. Anywhere else, the file-modification prohibition above still binds.
- Any test command needed to evaluate the change may be run. This is the sole exemption to the ban on state-changing shell actions.
- The reviewer may read every test the executor wrote. The executor is barred from reading anything under `tests/review_tests/` — the asymmetry is constitutional, since independent probes lose value the moment the audited party can study them.
- Independent testing is encouraged, not optional padding. Where the executor's test set leaves coverage gaps, exercises only happy paths, or skirts the failure modes the reviewer suspects, the reviewer SHOULD write probes that close those gaps and run them before forming the §4 verdict.
- `tests/review_tests/` MUST be entered in `.gitignore` and MUST NEVER enter the shared git index. Staging it would expose its contents to the executor via `git diff --cached`, defeating the asymmetry. Its contents live only in the reviewer's local working tree.

**Deliverable**:

- Internal context summary listing every file read, sufficient to defend each checklist verdict in §4.

**Exit condition**:

- All checklist-required material has been read; OR a specific material request has been delivered to the user and the session has paused.

## 4. Assessment

**Required actions**:

- Apply the checklist corresponding to the review type:
  - **G1** (Working Agreement §2.4): architecture consistency, interface compatibility, risk identification, test strategy.
  - **G2** (Working Agreement §2.6): consistency with approved plan, test coverage and passing, docs & indexes updated, no regressions.
  - **Audit**: the scope fixed at Intake.
- Each checklist item MUST receive an explicit answer with reasoning grounded in material read in §3.
- Formulate, for each failing item, one or more concrete Review Log entries of the form `- [Blocking|Non-blocking] [Concern|Question|Suggestion] <statement> — reasoning: <why>`. **Blocking** items preclude `APPROVED`; **Non-blocking** items are advisory and do not gate the verdict. These feed §5.
- Working-Agreement or subordinate-regulation breaches discovered during assessment MUST be recorded for the Violation section of the report in §5.

**Prohibitions**:

- Rubber-stamp judgments: ungrounded "LGTM", "approved", "no issues", or instant verdicts without specifics (`CLAUDE.md` §Anti-Rubber-Stamping).
- Fixing issues found during assessment. Fixes belong to Execution Authority and MUST be returned as Review Log entries in §5.
- Letting conversation length, user impatience, or token pressure shorten the evaluation.

**Deliverable (internal, consumed in §5)**:

- Per-item checklist assessment with verdict and reasoning.
- Review Log entries for every failing item (questions / concerns / suggestions with reasoning).
- Violation list for any Working-Agreement breach found.

**Exit condition**:

- Every checklist item has an explicit answer with reasoning, and Review Log entries are formulated for every failing item.

## 5. Verdict Delivery

**Required actions**:

- **Append to the plan file's `## Review Log` section** a new reviewer round block, headed `### <G1|G2> Round N — Reviewer — <APPROVED|REJECTED|NEEDS REVISION> — <YYYY-MM-DD HH:MM TZ>`, containing every Review Log entry formulated in §4. At G2 Round 1, the `## Review Log` section is absent (removed by the Post-G1 polish per `execution_authority.md` §3.1); the reviewer MUST create it afresh before appending. This is the authoritative channel for iterative discussion with the executor; the round number MUST be one higher than the most recent round present within the current gate.
- **Stage every change touched in this round** with `git add` — the modified plan file (now containing the new Review Log block) and any source files the executor altered since the last round, but explicitly **excluding** anything under `tests/review_tests/` (see §3.1). The git index then holds this round's shareable snapshot, so the next reviewer round reads the executor's response delta directly via `git diff` (working tree vs index).
- **Compose a structured report for the user**, with, in order:
  1. Opening line, exactly: `Verdict: <APPROVED|REJECTED|NEEDS REVISION> — Time: <YYYY-MM-DD HH:MM TZ>`.
  2. `Constitutional Violation` section (if any), listing every breach found in §4.
  3. Checklist with per-item verdict and reasoning.
  4. Final verdict line — exactly one of:
     - `APPROVED`
     - `REJECTED: <specific items>`
     - `NEEDS REVISION: <specific items>`
  5. (If `REJECTED` or `NEEDS REVISION`) A pointer to the Review Log round just appended, for executor action.
- Deliver the structured report to the user.

**Prohibitions**:

- Writing anywhere in the plan file other than inside the `## Review Log` section.
- Modifying, deleting, or reordering entries already present in `## Review Log` (executor responses included). The Log is append-only until the executor's final polish deletes the whole section.
- Writing to any file outside the plan file's Review Log. The §3.1 carve-out for `tests/review_tests/` belongs to Context Acquisition and is not exercised in §5.
- Delivering a verdict that contradicts the per-item assessments in §4.
- Vague Review Log entries in place of concrete, actionable items with reasoning.
- Driving an Execution Authority session directly. The verdict is returned to the user, who decides whether to launch a new Execution Authority session to act on it.

**Deliverable**:

- The appended Review Log round in the plan file.
- This round's full delta (plan + code) staged in the git index.
- The full structured report to the user.

**Exit condition**:

- Review Log round appended AND structured report delivered. The session exits Review Authority.

## 6. Violation Consequences

Any action that properly belongs to Execution Authority (source-file modification outside `tests/review_tests/`, commit, push, lint fix, or writing in the plan file outside its `## Review Log` section) voids the review verdict; the review MUST be re-issued by a fresh, independent Review Authority session before it may carry weight.

Stage violations (e.g., issuing a verdict before completing Assessment, accepting a review of one's own prior work, tampering with existing Review Log entries) likewise void the verdict; the review MUST restart from Intake in a new session.
