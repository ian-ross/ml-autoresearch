# Autoresearch Skill Set (review-only)

This directory contains the initial hierarchical Autoresearch Skill Set for ML Autoresearch. It is a review-only project-level skill directory, intentionally kept outside active `.pi/skills` and `.agents/skills` locations.

Do not symlink, install, or use these skills unattended until human review approves them.

The skill set uses progressive disclosure: start with `campaign-manager/SKILL.md`, then open only the focused skill needed for the current step and the linked project docs. The skills are written in the same `SKILL.md` convention as active skills, but this directory is documentation/review material until a human explicitly installs it.

## Autonomy Smoke Loop

Before unattended operation, run this short human-reviewed dry run:

1. Human selects one completed or synthetic-safe Research Problem workspace and a strict budget.
2. Agent opens `campaign-manager/SKILL.md`, reads `CONTEXT.md`, and proposes the next single Autonomous Research Iteration action without executing it.
3. Human checks the proposed delegation path and required docs.
4. Agent drafts one Experiment Proposal and one tiny Candidate Experiment or no-op observation plan, then stops for review.
5. Human verifies no covert workarounds, no direct Harness modifications, no direct Research Ledger edits, no arbitrary filesystem or network access, and no runtime weight downloads.
6. Agent records only approved Harness-owned events in a disposable ledger path.
7. Human reviews resulting artifacts and decides whether to install or revise the skill set.

Human review remains required before using these skills in unattended operation.
