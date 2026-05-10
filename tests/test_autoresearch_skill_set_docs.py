from pathlib import Path


SKILL_ROOT = Path("docs/autoresearch-skills")
REQUIRED_SKILLS = {
    "campaign-manager": [
        "Autonomous Research Iteration",
        "Campaign Pause Conditions",
        "human review",
    ],
    "proposal-writer": ["Experiment Proposal", "Comparison Target"],
    "candidate-implementer": ["Candidate Experiment Contract", "PROPOSAL.md"],
    "run-observer": ["Run", "Result", "get-best-runs"],
    "failure-classifier": ["Run Failure Classification", "Repair Candidate"],
    "evaluation-request-writer": ["Evaluation Request", "run-post-run-evaluation"],
    "research-note-writer": ["Research Note", "research-figures", "figure provenance"],
    "ledger-recorder": ["Research Ledger", "record-research-event"],
    "capability-request-writer": ["Capability Request", "not self-approving"],
    "campaign-report-writer": ["Campaign Report", "pause-campaign"],
    "pause-decider": ["Campaign Pause Conditions", "approved vocabulary"],
}
FORBIDDEN_GUARDRAILS = [
    "covert workarounds",
    "direct Harness modifications",
    "direct Research Ledger edits",
    "arbitrary filesystem",
    "network access",
    "runtime weight downloads",
]


def test_autoresearch_skill_set_lives_in_review_directory() -> None:
    readme = (SKILL_ROOT / "README.md").read_text()

    assert SKILL_ROOT.is_dir()
    assert not str(SKILL_ROOT).startswith(".pi/")
    assert not str(SKILL_ROOT).startswith(".agents/")
    assert "review-only" in readme
    assert "Do not symlink, install, or use these skills unattended until human review approves them." in readme
    assert "Autonomy Smoke Loop" in readme


def test_autoresearch_skills_have_progressive_disclosure_and_guardrails() -> None:
    for skill_name, required_phrases in REQUIRED_SKILLS.items():
        skill_path = SKILL_ROOT / skill_name / "SKILL.md"
        text = skill_path.read_text()

        assert text.startswith("---\nname:")
        assert "description:" in text
        assert "## Use" in text
        assert "## Read first" in text
        assert "## Guardrails" in text
        assert "CONTEXT.md" in text
        for phrase in required_phrases:
            assert phrase in text
        for guardrail in FORBIDDEN_GUARDRAILS:
            assert guardrail in text


def test_campaign_manager_delegates_to_focused_skills() -> None:
    text = (SKILL_ROOT / "campaign-manager" / "SKILL.md").read_text()

    for skill_name in REQUIRED_SKILLS:
        if skill_name != "campaign-manager":
            assert f"../{skill_name}/SKILL.md" in text
    assert "Do not continue automatically after a pause decision" in text
