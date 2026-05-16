Inside the Agent Control Boundary, the agent needs:

1. Read access

    - CONTEXT.md
    - docs/autoresearch-skills/
    - Harness docs: candidate contract, run lifecycle, evaluation request
      format, capability request format, campaign report format
    - Prior research-notes/
    - Prior candidate sources, if available
    - Run summaries/artifacts exposed by Harness-owned commands

    ⇒ read-only mounts


2. Write access to campaign workspace

    - New Candidate Experiment directories
    - PROPOSAL.md
    - Candidate README.md
    - Research Notes
    - Capability Requests
    - Evaluation Requests
    - Campaign Reports
    - Temporary scratch files

    ⇒ read/write mounts


3. Execute Harness-owned CLI

    - ml-autoresearch submit-candidate
    - ml-autoresearch run-candidate
    - ml-autoresearch list-runs
    - ml-autoresearch run-summary / get-run-summary
    - ml-autoresearch get-best-runs
    - ml-autoresearch run-post-run-evaluation
    - ml-autoresearch record-research-event
    - ml-autoresearch create-capability-request
    - ml-autoresearch record-campaign-report
    - ml-autoresearch pause-campaign

    ⇒ pre-installed in container image


4. No direct trusted infrastructure access

    - No Docker socket.
    - No secrets.
    - No cloud credentials.
    - No dataset write access.
    - No direct MLflow write token unless narrowly scoped and unavoidable.
    - No Harness code modification during autonomous campaign operation.

    ⇒ No mount of dataset directory.
    ⇒ No access to harness code.
    ⇒ External proxy for LLM access.


5. External web access

Reasonable, but risky. It enables architecture/literature lookup, but also
exfiltration if the sandbox can read sensitive files. So web access is only
reasonable if the mounted workspace contains nothing secret.

    ⇒ Allow external web access.


6. Run arbitrary Python code for restricted purposes

The agent should be allowed to run arbitrary Python inside its own container
for:

- parsing metrics/artifacts;
- making plots for Research Notes;
- summarizing CSV/JSON outputs;
- validating tensor-shape ideas on synthetic toy inputs;
- inspecting candidate source statically;
- generating candidate files;
- small non-authoritative sanity checks.

But it should not be allowed to use arbitrary Python to:

- train models outside the Harness;
- evaluate on raw datasets outside Harness-owned evaluation commands;
- bypass Candidate Experiment Contract limits;
- write unofficial result artifacts and treat them as Results;
- probe filesystem/environment for hidden capabilities;
- download pretrained weights into candidate code.

So the rule should be:

| Arbitrary Python is allowed for agent-side analysis and artifact preparation,
| but all authoritative Runs, Results, Post-Run Evaluations, ledger events, and
| resource use must go through Harness-owned CLI/API paths.

    ⇒ Pre-install Python in image
    ⇒ Pre-install common packages in image

