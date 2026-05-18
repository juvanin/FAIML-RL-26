# Axiom Skill Diffusion Design

**Date:** 2026-05-18
**Status:** Approved

## Problem

The proposed `humanized-ds-coding` skill offered useful utilities — scope thresholding,
plain-language result interpretation, baseline requirements for quantitative claims, and
inlined implementation style rules — but introducing it as a standalone skill would have
created trigger collisions and structural conflicts with the existing axiom skills.
`axiom-formalize` already contained a dangling reference to `humanized-ds-coding` in
Stage 5, confirming it was designed as a companion rather than a standalone layer.

## Decision

Diffuse the highest-value utilities directly into the three most relevant existing skills.
Leave `axiom-socratic` and `simplify-pass` unchanged. Drop the reporting template
(Background/Method/Results/Interpretation/Limitations), rigid section headers, and
named-pattern citation nudges entirely — these are flavoring, not function.

## What Changes

### `axiom-defend` — Scope Threshold Preamble

Add before the Core Mandates section:

```
## Scope Assessment (Run First)

Before applying the full defensive protocol, assess the complexity of the request:

- **Simple requests** — one-liner fixes, quick debug, minor utility snippets:
  apply naming and type hint rules only. Skip the full checklist.
- **Design-level requests** — pipelines, training loops, data splits, new modules,
  loss computation: apply the full defensive protocol below.

When in doubt, apply the full protocol.
```

**Why:** The full defensive protocol is necessary for design-level work but overkill for
a one-liner fix. Without this gate, the skill fires its entire checklist unconditionally,
which erodes trust and creates noise on simple requests.

---

### `axiom-formalize` — Scope Threshold + Inline Stage 5

**Change 1:** Same scope threshold preamble added before Stage 1. For simple math
questions, skip to a concise explanation rather than a full 4-stage formalization.

**Change 2:** Stage 5 currently reads:

> "write the implementation following the `humanized-ds-coding` style"

Replace with inline rules:

```
## Stage 5: Implementation (Unlocked After Approval)

Only after explicit approval, write the implementation with:
- Full type hints on all function signatures
- Google-style docstrings: one-line summary, why-paragraph, Args/Returns/Raises/Defensive Notes
- Assert statements at every tensor shape transition (pairs with axiom-defend)
- Complexity note: state time and memory complexity, and any vectorization or
  caching choices made — 2–4 sentences or a short bullet list, not an essay
```

**Why:** The dangling reference to `humanized-ds-coding` would silently break once that
skill is not installed. Inlining removes the dependency and makes Stage 5 self-contained.

---

### `axiom-falsify` — Plain-Language Interpretation + Baseline Requirement

Append two new sections after the Claim Escalation Ladder:

```
## Plain-Language Result Interpretation

When code produces output — metrics, predictions, plots, tables — the terminal result
is necessary but not sufficient. After showing verified output, interpret what it means
in plain language. Avoid jargon unless it has been defined.

❌ "The model achieves 0.87 AUC-ROC."
✅ "The model correctly ranks a positive case above a negative one about 87% of the
    time — strong for a dataset with this class imbalance."

This is not a style preference. An uninterpreted number is an unverified claim about
meaning.

## Baseline Requirement

Every quantitative result must be paired with a reference point before it can be
treated as meaningful. Acceptable baselines: majority-class accuracy, random policy
return, prior published result, or the same model without the proposed change.

❌ "Training return: 450"
✅ "Training return: 450 (random policy baseline: 18; solved threshold: 3500)"

A number without a baseline is a guess dressed as a result. The claim escalation
ladder applies to baselines too — state the baseline source.
```

**Why:** `axiom-falsify` already polices overconfident language and requires terminal
proof. These additions extend that discipline from "is the number correct?" to "does the
number mean anything?" — closing the gap between verification and comprehension.

---

## What Does Not Change

| Skill | Reason |
|---|---|
| `simplify-pass` | Already covers naming, imports, and Google-style docstrings in Passes 2–4 |
| `axiom-socratic` | Generates no code or result output; style rules do not apply |

## What Is Excluded from `humanized-ds-coding`

| Element | Reason for exclusion |
|---|---|
| Background/Method/Results/Interpretation/Limitations template | Too rigid for conversational ML responses; adds structure without function |
| "Name the established pattern" nudge | Tends toward forced or hallucinated citations on routine tasks |
| Modular structure rules | Already covered by `axiom-defend` (pure functions) and `simplify-pass` |
| Naming conventions | Already covered by `simplify-pass` Pass 3 |
| Import grouping | Already covered by `simplify-pass` Pass 2 |

## Implementation Scope

Three files to edit:
1. `.claude/skills/axiom-defend/SKILL.md` — insert scope threshold block
2. `.claude/skills/axiom-formalize/SKILL.md` — insert scope threshold block, replace Stage 5 reference
3. `.claude/skills/axiom-falsify/SKILL.md` — append two new sections after Claim Escalation Ladder
