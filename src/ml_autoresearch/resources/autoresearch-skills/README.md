# Autoresearch Skill Set

This directory is the single source of truth for the packaged Autoresearch Skill Set. `prepare-agent-boundary` installs these skills into `agent-work/.pi/skills/` so the inner agent can use `campaign-manager` and the focused Autoresearch skills inside the Agent Control Boundary.

For human browsing from the documentation tree, `docs/autoresearch-skills` is a relative symlink to this directory. Do not create a second copy of these skills.

The skill set uses progressive disclosure: start with `campaign-manager/SKILL.md`, then open only the focused skill needed for the current step and the linked project docs. Changes to these skills still require human review because they affect autonomous agent behavior.

## Autonomy Smoke Loop

Before unattended operation, run this human-reviewed dry run:

1. Human selects one completed or synthetic-safe Research Problem workspace and a strict budget.
2. Agent opens `campaign-manager/SKILL.md`, reads `CONTEXT.md`, and proposes the next single Autonomous Research Iteration action without executing it.
3. Human checks the proposed delegation path and required docs.
4. Agent drafts one Experiment Proposal and one tiny Candidate Experiment or no-op observation plan, then stops for review.
5. Human verifies no covert workarounds, no direct Harness modifications, no direct Research Ledger edits, no arbitrary filesystem or network access, and no runtime weight downloads.
6. Agent records only approved Harness-owned events in a disposable ledger path.
7. Human reviews resulting artifacts and decides whether the skill set is ready for unattended use in the Agent Control Boundary.
