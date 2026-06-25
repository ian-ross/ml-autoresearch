# Autoresearch Skill Set

This directory is the source of truth for the packaged Autoresearch Skill Set. `prepare-agent-boundary` installs these skills into `agent-work/.pi/skills/` so the inner agent can use `campaign-manager` and focused Autoresearch skills inside the Agent Control Boundary.

For browsing from the documentation tree, `docs/autoresearch-skills` is a relative symlink here. Do not create a second copy.

The skill set uses progressive disclosure: start with `campaign-manager/SKILL.md`, then open only the focused skill needed for the current step and linked project docs. Changes still require human review because they affect autonomous agent behavior.

## Autonomy Smoke Loop

Before unattended use, run this human-reviewed dry run:

1. Human selects one completed or synthetic-safe Research Problem workspace and a strict budget.
2. Agent opens `campaign-manager/SKILL.md`, reads `CONTEXT.md`, and proposes the next single Autonomous Research Iteration action without executing it.
3. Human checks the proposed delegation path and required docs.
4. Agent drafts one Experiment Proposal and one tiny Candidate Experiment or no-op observation plan, then stops for review.
5. Human verifies there are no covert workarounds, direct Harness modifications, direct Research Ledger edits, arbitrary filesystem or network access, or runtime weight downloads.
6. Agent records only approved Harness-owned events in a disposable ledger path.
7. Human reviews artifacts and decides whether the skill set is ready for unattended use in the Agent Control Boundary.
