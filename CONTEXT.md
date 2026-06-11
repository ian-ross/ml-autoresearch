# ML Autoresearch

ML Autoresearch is a safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments. The initial Research Problem is Ground-Camera Contrail Detection, but the system is intended to support other Research Problems later.

## Language

**ML Autoresearch**:
A safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments.
_Avoid_: Contrail runner, Docker runner, Pi runner

**Research Problem**:
A target ML problem that ML Autoresearch explores through Candidate Experiments, including its data modality, prediction target, evaluation metrics, augmentation semantics, and constraints.
_Avoid_: Task, use case, project

**Research Problem Spec**:
A trusted Harness-side registration for one Research Problem's concrete capabilities, such as dataset adapters, input modes, prediction targets, metrics, augmentation policies, auxiliary targets, allowed losses, reporting templates, and figure selectors. A Research Problem Spec is trusted infrastructure, not Candidate Experiment authority.
_Avoid_: Candidate plugin, candidate-provided dataset adapter, arbitrary task config

**Research Problem Brief**:
Advisory, versioned documentation declared by a filesystem Research Problem package to support progressive disclosure of context such as problem overview, domain/data notes, literature, baselines, and modeling suggestions. The Research Problem Brief informs agents and humans but is not the normative machine-checkable execution contract unless a document is explicitly marked required by the interface.
_Avoid_: Hidden contract, prompt-only spec, candidate authority

**Filesystem Research Problem Package**:
A trusted local filesystem package that exposes a checked Research Problem Spec provider for one Research Problem outside the reusable Harness package.
_Avoid_: Built-in Harness problem package, candidate plugin, PyPI plugin marketplace

**Research Problem Spec Registry**:
The trusted Harness-owned seam where Research Problem Specs are registered so the reusable Harness can discover problem-specific capabilities without hard-coding every Research Problem into core infrastructure.
_Avoid_: Plugin marketplace, candidate extension point, dynamic untrusted registry

**Problem Support Library**:
A reusable trusted library of segmentation, imaging, metric, target-construction, and reporting helpers used by Research Problem Specs and the Harness when multiple Research Problems share implementation patterns. It provides building blocks but does not decide Research Problem policy by itself.
_Avoid_: Candidate helper library, policy registry, generic plugin API

**Research Problem Repository**:
A future trusted package boundary for Research Problem-specific definitions such as dataset adapters, input modes, prediction targets, metrics, augmentation policies, auxiliary targets, allowed losses, reporting templates, and figure selectors, separated from reusable ML Autoresearch infrastructure after the proving-ground loop matures.
_Avoid_: Fork of the Harness, candidate repository, candidate-provided plugin

**Research Loop**:
The closed cycle in which Candidate Experiments are proposed, the system runs them, Results are evaluated, and the next Candidate Experiment is informed by prior Results. In early Human-Guided Research Iterations, humans may make proposal decisions before the Pi-agent proposal loop is automated.
_Avoid_: Pipeline when referring to the iterative research cycle

**Human-Guided Research Iteration**:
An early Research Loop iteration where a human chooses or edits the next Candidate Experiment, uses the Harness to run it, and inspects Results before deciding the next step. This bridges the completed tracer-bullet Harness and a later autonomous Pi-agent proposal loop.
_Avoid_: Manual testing when referring to learning-oriented research iteration

**Autonomous Research Iteration**:
A Research Loop iteration where the agent reviews prior Results and Research Notes, proposes and implements one Candidate Experiment or a bounded Experiment Batch within the Candidate Experiment Contract, submits Runs, observes Results, writes a Research Note, and chooses the next step without changing the Research Problem or Harness policy.
_Avoid_: Fully autonomous science, unrestricted agent control

**Autoresearch Skill Set**:
A hierarchical set of focused agent skills with progressive disclosure, where a campaign-manager skill orchestrates proposal, candidate implementation, Run observation, failure classification, Evaluation Requests, Research Notes with figures, Research Ledger updates, Capability Requests, Campaign Reports, and pause decisions.
_Avoid_: Monolithic prompt, implicit agent workflow

**Autonomy Smoke Loop**:
A short autonomous run using the current narrow Candidate Experiment Contract to test disciplined autonomous behavior, including proposals, contract compliance, bounded repairs, failure classification, Research Notes with figure provenance, Research Ledger events, Capability Requests, Campaign Reports, and pause behavior before broader contract expansion.
_Avoid_: Production autonomy, multi-week campaign, metric-improvement gate

**Autonomy Step**:
One outer-Harness orchestration cycle that refreshes the Agent Control Boundary, invokes the agent once, ingests exactly one primary handoff outcome, updates canonical state, and selects the next action for an Autonomous Research Iteration.
_Avoid_: Full autonomous iteration, agent session, multi-action batch, manual test step

**Experiment Batch**:
A bounded set of related Candidate Experiments proposed together to compare independent variants under one research hypothesis.
_Avoid_: Unbounded search, sweep when the variants are not tied to a single hypothesis

**Research Progress**:
Sustained Research Loop value, combining best validated Result improvement with hypothesis-driven reduction of uncertainty about which model families, input modes, losses, data policies, and training policies merit further investment.
_Avoid_: Leaderboard-only optimization, metric paperclip maximization

**Capability Request**:
A structured, human-gated request from the agent to expand Harness-owned contract surface, approved resources, or operational policy when current capabilities block useful Research Progress; Harness changes happen only in a separate human-supervised process outside the autonomous loop.
_Avoid_: Self-approved contract change, arbitrary candidate authority, covert workaround

**Research Note**:
A human- and agent-readable record of what a Run or comparison tested, what Result it produced, what qualitative behavior was observed, which provenance-tracked figures illustrate the behavior, and what decision follows from it.
_Avoid_: Log file, paper, raw metrics dump

**Research Figure**:
A human-readable visual artifact embedded in or referenced by a Research Note, with explicit provenance linking it to the Run or Post-Run Evaluation artifacts it summarizes.
_Avoid_: Untraceable screenshot, cherry-picked image without source

**Research Ledger**:
An append-only structured research-memory event log recording proposals, Candidate Experiments, Runs, Results, Research Notes, decisions, failed attempts, contract features used, and pending Capability Requests across the Research Loop.
_Avoid_: Raw metrics dump, human-only notebook, mutable spreadsheet as canonical memory

**Experiment Proposal**:
A short rationale for the next Candidate Experiment or Experiment Batch, written before code generation, that states the hypothesis, expected effect, implementation sketch, constraints, declared comparison target, and success criteria. The proposal is the bridge between reviewing prior Research Notes and generating Candidate Experiment code for the Harness.
_Avoid_: Candidate Experiment when referring only to the rationale before runnable code exists

**Candidate README**:
A Candidate Experiment document written with the implementation to summarize the Model Architecture, manifest choices, contract assumptions, and known limitations.
_Avoid_: Experiment Proposal, Research Note

**Comparison Target**:
The declared baseline or control that a Candidate Experiment or Experiment Batch is primarily intended to compare against, which may be the global best Run, a family baseline, or a direct parent Run.
_Avoid_: Implicit leaderboard-only baseline

**Candidate Experiment**:
An agent-proposed, runnable ML research package containing the model architecture and allowed configuration needed for the runner to evaluate one research idea.
_Avoid_: Candidate, architecture, package, submission when referring to the whole unit

**Repair Candidate**:
A distinct Candidate Experiment version created to fix a candidate bug in an earlier submitted Candidate Experiment while preserving repair lineage and auditability.
_Avoid_: Overwritten candidate, silent resubmission

**Candidate Experiment Contract**:
The allowlisted interface through which a Candidate Experiment can express research variation without receiving unsafe authority over training, data loading, filesystem access, network access, Docker, or MLflow writes.
_Avoid_: Free-form plugin API, arbitrary experiment code

**Initial Flexibility Envelope**:
The v1 set of research variations exposed by the Candidate Experiment Contract: model architecture, input mode, output form, loss selection, bounded training knobs, augmentation/data policy, and pretrained weight requests.
_Avoid_: Minimal contract when that implies low research expressiveness

**Capability Slice**:
A small, Harness-owned expansion of the Candidate Experiment Contract that unlocks a meaningful family of Candidate Experiments without granting broad new candidate authority.
_Avoid_: Grab-bag feature, arbitrary plugin surface

**Wall-Clock Budget Policy**:
The Harness-owned policy limiting Run duration, expected to be adjustable so early research can prefer many cheap experiments before longer training runs.
_Avoid_: Fixed training duration baked into Candidate Experiments

**Research Campaign**:
A bounded autonomous research effort made of many Autonomous Research Iterations, governed by campaign-level, batch-level, and per-Run budgets plus pause conditions and reporting cadence.
_Avoid_: Unbounded agent session, single Run

**Campaign Pause Condition**:
A Harness-owned condition that pauses an autonomous Research Campaign for human review, such as budget exhaustion, repeated failures, stalled Research Progress, too many pending Capability Requests, storage risk, or scheduled check-in.
_Avoid_: Agent whim, silent campaign termination

**Campaign Report**:
A human-readable status artifact for a Research Campaign, produced on a fixed cadence or in response to events such as pause conditions, significant improvements, repeated failures, or high-priority Capability Requests.
_Avoid_: Raw run log, metric dump without interpretation

**Resource Budget Policy**:
The Harness-owned policy constraining compute resources for Runs and Research Campaigns, including GPU assignment, CPU memory, shared memory, process limits, concurrency, model/training bounds that reduce GPU out-of-memory risk, and bounded retry behavior after resource failures.
_Avoid_: Candidate-selected hardware access, best-effort resource use

**Resource Failure**:
A Run failure caused by exhausting Harness-governed compute resources, such as GPU memory, CPU memory, shared memory, process limits, or wall-clock budget.
_Avoid_: Model-quality failure, validation metric regression

**Run Failure Classification**:
A post-failure label distinguishing candidate bugs, contract violations, Resource Failures, Harness failures, and bad research results so the autonomous loop can choose the correct next action.
_Avoid_: Generic failed status without cause

**Resolved Manifest**:
The Harness-owned record of the Candidate Experiment manifest after validation, normalization, and Harness defaults have been applied for a specific Run.
_Avoid_: Source manifest, candidate configuration

**Harness**:
The trusted outer implementation that owns training loops, data loading, validation, execution policy, artifact persistence, and approved parameterized variations.
_Avoid_: Candidate code, agent code

**Model Architecture**:
The neural network design inside a Candidate Experiment.
_Avoid_: Candidate experiment when referring only to the network design

**Architecture Helper**:
A Candidate Experiment helper module that defines layers, blocks, or model-composition code used by the Model Architecture without performing data loading, training, loss computation, evaluation, artifact writing, filesystem or environment inspection, subprocess execution, networking, or dependency management.
_Avoid_: General helper, utility module when it can smuggle non-architecture behavior

**Run**:
One execution attempt of a Candidate Experiment by the Candidate Experiment Runner.
_Avoid_: Experiment when referring only to execution

**Result**:
The metrics and artifacts produced by a Run, with final-epoch metrics distinguished from best-validation metrics.
_Avoid_: Run when referring only to observed outputs

**Post-Run Evaluation**:
A Harness-owned evaluation pass that reloads a completed Run's copied Candidate Experiment and persisted model artifact to compute additional metrics or diagnostic artifacts without retraining.
_Avoid_: New Run, Candidate Experiment, ad hoc inference script

**Evaluation Request**:
A lightweight pre-evaluation record stating the target Run, approved Post-Run Evaluation mode, diagnostic question, expected decision impact, bounded diagnostic parameters, and artifact or resource budget.
_Avoid_: Experiment Proposal, ad hoc diagnostic command

**Candidate Experiment Runner**:
The trusted subsystem that validates, executes, and records agent-proposed Candidate Experiments without exposing host secrets, Docker, GPU control, or unapproved infrastructure authority to the agent.
_Avoid_: Pi runner, Docker runner, contrail runner

**Agent Control Boundary**:
The boundary that prevents the agent from directly accessing host shell, Docker, GPU or cluster control, secrets, cloud credentials, or unrestricted infrastructure/network resources, while allowing explicitly configured read-only Research Problem data access when policy permits.
_Avoid_: Sandbox when referring specifically to agent permissions

**Agent Workspace**:
The writable area inside the Agent Control Boundary where the agent drafts Candidate Experiments and creates autonomous-loop artifacts before Harness ingestion.
_Avoid_: History, Harness workspace, run directory

**Research History**:
The read-only prior research material exposed inside the Agent Control Boundary for review, including prior Candidate Experiments, Runs, Research Notes, Research Ledger events, and trusted docs.
_Avoid_: Workspace, scratch area

**Agent Reference Snapshot**:
A setup-generated, read-only directory exposed inside the Agent Control Boundary containing copies of root-level canonical reference files that cannot be mounted individually, such as `CONTEXT.md` and `EXPERIMENT_INDEX.md`.
_Avoid_: Canonical source file, editable reference

**Candidate Submission Queue**:
A Harness-ingested handoff area where the agent places finalized Candidate Experiment submissions that are ready for validation and execution by the Harness.
_Avoid_: Draft candidates directory, runs queue

**Agent Handoff Ingestion**:
The Harness-owned act of copying exactly one primary handoff outcome from the Agent Workspace to its canonical destination, marking the source as ingested, and recording an auditable Research Ledger event before any optional Run or evaluation execution.
_Avoid_: Candidate execution, direct agent scheduling, moving workspace files

**Candidate Submission Preparation**:
A Harness-owned packaging step inside the Agent Control Boundary that statically validates a Candidate Experiment draft, requires its Experiment Proposal, copies it into the Candidate Submission Queue, and writes submission metadata without importing model code or running smoke tests.
_Avoid_: Run submission, smoke test, training

**Candidate Execution Boundary**:
The boundary that contains untrusted Candidate Experiment code while it is imported, smoke-tested, trained, and evaluated.
_Avoid_: Agent boundary, static validation

**Approved Weight Artifact**:
An audited pretrained weight artifact made available to the Harness under a stable identifier for use by approved Candidate Experiments.
_Avoid_: Download URL, arbitrary checkpoint path, candidate-managed weights

**Pretrained Weight Request**:
A request from the agent to add a new Approved Weight Artifact, including source, license, intended use, and audit information.
_Avoid_: Runtime weight download, candidate-managed checkpoint fetch

**Ground-Camera Contrail Detection**:
The initial Research Problem and infrastructure proving ground for ML Autoresearch, focused on binary semantic segmentation of contrail pixels vs non-contrail pixels in ground-camera imagery.
_Avoid_: The project, the product

**Satellite Contrail Detection**:
A future Research Problem for ML Autoresearch, expected to reuse the autoresearch infrastructure after it is proven on Ground-Camera Contrail Detection.
_Avoid_: Current GVCCS Research Problem

**GVCCS Dataset**:
The whole-sky-camera training dataset for the Ground-Camera Contrail Detection Research Problem, published at https://zenodo.org/records/16612390.
_Avoid_: Contrail dataset when a specific dataset identity is needed

**Working Validation Split**:
An agent-visible validation split used for iterative Research Loop feedback and model comparison during autonomous exploration.
_Avoid_: Final test set, locked evaluation

**Locked Evaluation Split**:
A human-gated evaluation split withheld from autonomous iteration and used for milestone checks to reduce validation overfitting risk.
_Avoid_: Working validation, agent-visible leaderboard

**Camera Domain Shift**:
The expected distribution difference between GVCCS whole-sky-camera training data and likely downstream conventional ground-camera imagery.
_Avoid_: Treating GVCCS validation as deployment validation

**Contrail Mask**:
A binary semantic segmentation label marking each pixel as contrail or non-contrail.
_Avoid_: Classification label, detection box

**Auxiliary Target**:
A Harness-derived per-pixel training target used for an auxiliary loss, not a separate primary prediction target. An Auxiliary Target is judged by whether it is suitably conditioned to guide training for the Research Problem; it does not need to be a semantically pure transformation such as an exact skeleton or exact boundary.
_Avoid_: Primary label, end-user prediction, exact annotation

**Line Target**:
An Auxiliary Target derived from the Contrail Mask to emphasize thin centerline-like contrail structure.
_Avoid_: Hough-space target, line detection output

**Boundary Target**:
An Auxiliary Target derived from the Contrail Mask to emphasize contrail edge geometry.
_Avoid_: Object-detection boundary, separate segmentation class

**Target Frame**:
The frame whose Contrail Mask a Candidate Experiment predicts.
_Avoid_: Label frame, center image

**Frame Sequence**:
A temporally ordered group of GVCCS Dataset frames from the same camera scene, inferred by the Harness from timestamp-like filenames with 30-second inter-frame spacing.
_Avoid_: Video when referring to inferred image-frame groups

**Data Policy**:
A Harness-owned set of choices controlling how data examples are selected, ordered, transformed, and presented to a Candidate Experiment.
_Avoid_: Candidate data loader, arbitrary input pipeline

**Sampling Policy**:
A candidate-selectable, Harness-owned part of the Data Policy controlling training example order, initially limited to deterministic shuffle and sequential order.
_Avoid_: Custom sampler code, candidate data loader

**Frame Selection Policy**:
A Research Problem-owned, candidate-selectable part of the Data Policy controlling which Target Frames are eligible for training and validation before ordering and batching. For Ground-Camera Contrail Detection, this includes using all Target Frames or only temporal-eligible center Target Frames with complete neighboring frames inside one Frame Sequence.
_Avoid_: Padding policy, candidate sampler, arbitrary frame filter

**Augmentation Policy**:
A Research Problem-owned, candidate-selectable part of the Data Policy controlling approved Harness-executed transforms applied to selected training examples.
_Avoid_: Custom augmentation code, candidate transforms, system-wide augmentation assumption

**Prediction Sample Policy**:
A Harness-owned, Run-selectable policy controlling which qualitative prediction examples are saved for Run inspection, initially including first-N and adjacent-plus-scattered sample selection with probability heatmaps.
_Avoid_: Ad hoc screenshots, cherry-picked examples

**Input Mode**:
A Harness-owned choice of what image or video tensor is provided to the model for a Target Frame.
_Avoid_: Candidate data loader, arbitrary input pipeline

**Single-Frame RGB Input**:
An Input Mode where the model receives one RGB image and predicts the Contrail Mask for that same Target Frame.
_Avoid_: Image classification input

**Centered Temporal RGB Clip Input**:
An Input Mode where the model receives multiple RGB frames around a Target Frame and predicts the Contrail Mask for that Target Frame.
_Avoid_: Video-level prediction, arbitrary video segment

## Relationships

- **ML Autoresearch** explores one or more **Research Problems**.
- **Ground-Camera Contrail Detection** is the first **Research Problem** for **ML Autoresearch**.
- **Ground-Camera Contrail Detection** is used to prove the autoresearch infrastructure before applying it to future Research Problems such as **Satellite Contrail Detection**.
- A **Research Problem Spec** is the near-term trusted registration unit for a **Research Problem** before mature definitions move into separate **Research Problem Repositories**.
- A **Research Problem Brief** provides advisory context alongside a checked **Research Problem Spec** without replacing the normative machine-checkable contract.
- A **Filesystem Research Problem Package** is the source-location boundary for trusted Research Problem-specific code; the semantic boundary is the checked **Research Problem Spec** it returns.
- The **Research Problem Spec Registry** separates reusable **Harness** infrastructure from trusted problem-specific capability registration.
- **Research Problem Specs** are Harness-side trusted registrations, not **Candidate Experiment** plugins or authority to bypass the **Candidate Experiment Contract**.
- The **Problem Support Library** provides reusable trusted building blocks for **Research Problem Specs** without becoming an independent policy boundary or Candidate-facing plugin API.
- Mature **Research Problem** definitions are expected to move into trusted **Research Problem Repositories** that register problem-specific capabilities with reusable ML Autoresearch infrastructure.
- **Ground-Camera Contrail Detection** remains the only real registered **Research Problem** initially; other problems should wait until the GVCCS proving-ground loop justifies extraction.
- Temporal input should be implemented after the **Research Problem Spec Registry** seam so GVCCS-specific **Frame Sequence**, **Target Frame**, sampling, augmentation, and reporting semantics are registered problem capabilities rather than ad hoc core **Harness** behavior.
- **Ground-Camera Contrail Detection** uses the **GVCCS Dataset** for training.
- **Camera Domain Shift** is a known limitation of using the **GVCCS Dataset** for models likely to be tried on conventional ground-camera imagery.
- Evaluation on non-GVCCS camera data is a separate exercise outside the initial **ML Autoresearch** loop.
- The prediction target for **Ground-Camera Contrail Detection** is a **Contrail Mask** for a **Target Frame**.
- A **Frame Sequence** groups temporally adjacent GVCCS **Target Frames** for sampling and qualitative diagnostics without implying that a single-frame Candidate Experiment receives temporal input.
- For the **GVCCS Dataset**, consecutive frames within a **Frame Sequence** are exactly 30 seconds apart; any larger timestamp gap starts a different **Frame Sequence**.
- **Line Target** and **Boundary Target** are optional **Auxiliary Targets** derived from the **Contrail Mask** by the **Harness**.
- **Auxiliary Targets** are used for auxiliary training losses; primary evaluation remains based on the **Contrail Mask** prediction.
- **Data Policy** includes **Frame Selection Policy**, **Sampling Policy**, **Augmentation Policy**, and qualitative **Prediction Sample Policy**.
- **Frame Selection Policy**, **Sampling Policy**, and **Augmentation Policy** are candidate-selectable only through Harness-executed allowlists, not custom candidate data-loading or transform code.
- **Frame Selection Policy** and **Augmentation Policy** choices are scoped to a **Research Problem** because valid frame eligibility and transforms depend on data modality and dataset semantics.
- Candidate Experiments declare **Sampling Policy** under `data.sampling_policy` in the manifest; older manifests without it resolve to the previous sequential behavior for compatibility.
- Candidate Experiments may declare **Frame Selection Policy** under `data.frame_selection_policy` in the manifest; `single_frame_rgb` defaults to `all_target_frames`, while `centered_temporal_rgb_clip` defaults to and requires `temporal_eligible_center`.
- Initial **Sampling Policy** choices affect training example order; validation order remains stable for reproducible metrics and qualitative diagnostics.
- **Temporal-eligible center** frame selection excludes sequence-boundary frames and never pads, duplicates, or crosses **Frame Sequence** gaps.
- **Prediction Sample Policy** is Harness-owned, selected at Run time rather than by Candidate Experiments, and affects qualitative Result artifacts, not model training.
- The adjacent portion of the initial **Prediction Sample Policy** selects stride-1 consecutive **Target Frames** from validation **Frame Sequences** with non-empty **Contrail Masks**, spaced across eligible sequences.
- The scattered portion of the initial **Prediction Sample Policy** is positive-biased while retaining a small negative slice to inspect false positives on empty **Contrail Masks**.
- **Single-Frame RGB Input** and **Centered Temporal RGB Clip Input** are v1 **Input Modes** for **Ground-Camera Contrail Detection**.
- **ML Autoresearch** works by sustaining a **Research Loop**.
- A **Research Loop** explores a **Research Problem** through many **Candidate Experiments**.
- **Human-Guided Research Iterations** are early **Research Loop** iterations used before **Autonomous Research Iterations** are implemented.
- **Autonomous Research Iterations** allow the agent to conduct Candidate Experiment proposal, implementation, submission for Harness execution, observation, note-writing, and next-step selection within fixed Harness and Research Problem boundaries.
- An **Autoresearch Skill Set** should use focused skills with progressive disclosure, orchestrated by a campaign-manager skill rather than one monolithic prompt.
- An **Autonomy Smoke Loop** uses the current narrow **Candidate Experiment Contract** to test autonomous behavior and whether the agent creates **Capability Requests** instead of attempting covert workarounds.
- An **Autonomous Research Iteration** normally proposes one **Candidate Experiment**, but may propose an **Experiment Batch** when parallel variants are bounded and tied to one research hypothesis.
- An **Experiment Batch** is limited by Harness policy for maximum parallel submissions.
- **Research Progress** is judged by both best validated Result improvement and hypothesis-driven exploration, not by leaderboard improvement alone.
- Stalled **Research Progress** is assessed by a combination of metric plateau and structured agent self-assessment about recent hypothesis value, repetition, and blocked next steps.
- A **Research Campaign** is governed by hierarchical budgets, including campaign-level, batch-level, and per-Run limits.
- A **Research Campaign** pauses for human review when a **Campaign Pause Condition** is reached.
- A **Research Campaign** produces **Campaign Reports** on fixed cadence and in response to important events.
- **Resource Budget Policy** is Harness-owned and includes GPU assignment, CPU memory, shared memory, process limits, concurrency, model/training bounds that reduce GPU out-of-memory risk, and bounded retry behavior after **Resource Failures**.
- A GPU out-of-memory error is a **Resource Failure**; the Harness may perform bounded retry with a smaller effective batch size before failing the Run with a clear resource failure reason.
- When the Harness lowers batch size after a **Resource Failure**, the **Resolved Manifest** or Run metadata records both requested and effective batch size so Result comparisons remain interpretable.
- After a failed Run, the autonomous loop records a **Run Failure Classification** before deciding whether to fix the candidate, create a Capability Request, propose a smaller variant, pause for human review, or treat the outcome as research evidence.
- During an **Autonomous Research Iteration**, the agent may use existing **Candidate Experiment Contract** choices but must create a human-gated **Capability Request** before expanding Harness-owned contract surface, approved resources, or operational policy.
- During autonomous operation, the agent may create and submit Candidate Experiments, request Research Ledger updates through Harness-owned commands, create Research Notes, create Research Figures, and create Capability Requests; it must not modify Harness code, tests, or trusted infrastructure docs.
- A **Research Ledger** is the structured research memory for the autonomous loop, while **Research Notes** provide human-readable interpretation.
- The initial **Research Ledger** storage format is append-only `research-ledger.jsonl`.
- **Research Ledger** events are appended through a Harness-owned command or API rather than direct agent file writes.
- The agent does not create generic **Research Ledger** event requests; it creates explicit artifacts such as Candidate Experiment submissions, Research Notes, Capability Requests, Evaluation Requests, and Campaign Reports, from which the Harness records validated ledger events.
- Harness-owned commands that create durable research artifacts append validated **Research Ledger** events by default, regardless of whether a human or agent invoked them.
- Initial **Research Ledger** event types include proposal created, candidate created or submitted, Run started or completed or failed, Research Note written, Capability Request created, Campaign Report written, and Research Campaign paused.
- Later **Research Ledger** event types may include Research Figure created, resource retry, comparison recorded, and budget updated.
- A **Research Note** records the observed outcome, qualitative **Research Figures**, and decision from a **Run** or comparison of **Runs**.
- A **Research Figure** must record provenance through source Run IDs or Post-Run Evaluation IDs and source artifact paths.
- An **Experiment Proposal** uses prior **Research Notes**, the **Research Ledger**, and constraints to justify the next **Candidate Experiment** before runnable code is generated.
- Each **Autonomous Research Iteration** requires an **Experiment Proposal** before Candidate Experiment code is generated or submitted.
- Each **Experiment Proposal** is stored with its Candidate Experiment source and indexed by the **Research Ledger**.
- A **Candidate README** summarizes implementation details and is distinct from the pre-implementation **Experiment Proposal** and the post-Run **Research Note**.
- Each **Experiment Proposal** declares a **Comparison Target** and success criteria; global-best comparison is still reported but is not always the primary control.
- **ML Autoresearch** uses a **Candidate Experiment Runner** to execute agent-proposed **Candidate Experiments** safely.
- A **Research Problem** is explored through many **Candidate Experiments**.
- A **Candidate Experiment** conforms to a **Candidate Experiment Contract**.
- A submitted **Candidate Experiment** is not overwritten; bug fixes are submitted as **Repair Candidates** with distinct source and repair lineage.
- Initial autonomous policy allows at most two **Repair Candidates** per original proposal before the agent abandons the line, writes a Research Note, creates a Capability Request, or pauses as appropriate.
- A **Repair Candidate** preserves the original hypothesis and Comparison Target; changes to the scientific idea require a new Experiment Proposal and Candidate Experiment lineage.
- A **Candidate Experiment** may have a **Resolved Manifest** for each accepted **Run**.
- A **Resolved Manifest** records what the **Harness** actually ran, not just what the Candidate Experiment requested.
- A **Candidate Experiment** contains one **Model Architecture**.
- Candidate helper code is limited to **Architecture Helpers** used by the **Model Architecture**.
- A **Candidate Experiment** may produce zero or more **Runs**.
- A **Run** produces at most one **Result**.
- A completed **Run** may have zero or more **Post-Run Evaluations**.
- During autonomous operation, the agent may invoke approved, bounded, Harness-owned **Post-Run Evaluations** on prior Runs without creating new Candidate Experiments.
- During autonomous operation, each **Post-Run Evaluation** requires an **Evaluation Request** before execution.
- An **Evaluation Request** may choose bounded Harness-approved diagnostic parameters such as thresholds, threshold sweep bounds, evaluation batch size, artifact counts, and failure buckets.
- A **Post-Run Evaluation** uses the completed **Run**'s copied **Candidate Experiment**, **Resolved Manifest**, and persisted model artifact rather than retraining or creating a new **Candidate Experiment**.
- A **Result** distinguishes final completed epoch metrics from best-validation metrics so research decisions do not confuse final model state with peak observed validation behavior.
- For **Ground-Camera Contrail Detection**, the initial best-validation metric is Dice over the **Contrail Mask** on the **Working Validation Split**.
- Autonomous exploration uses the **Working Validation Split** for iteration, while **Locked Evaluation Split** access is reserved for human-gated milestone checks.
- **Locked Evaluation Split** Results may be summarized to the agent in Campaign Reports after human-triggered milestone checks, but access frequency remains limited to reduce overfitting risk.
- Initial best-validation reporting does not imply checkpoint restoration, but best-epoch model artifact persistence is expected soon for evaluation beyond the original Run.
- The **Initial Flexibility Envelope** is part of the **Candidate Experiment Contract** from the beginning.
- The **Initial Flexibility Envelope** includes model architecture, input mode, output form, loss selection, bounded training knobs, candidate-selectable **Data Policy**, and pretrained weight requests.
- **Wall-Clock Budget Policy** is intentionally adjustable and may start small to encourage many cheap architecture-search Runs.
- The **Harness** owns training loops, data loading, validation, execution policy, artifact persistence, and approved parameterized variations.
- The **Candidate Experiment Contract** can expose Harness-owned parameters for model architecture, input modes, output forms, losses, optimizer choices, training budgets, augmentation/data policy, and pretrained weight availability.
- The **Candidate Experiment Contract** should expand through **Capability Slices** that unlock meaningful experiment families while preserving Harness ownership.
- Expected near-term **Capability Slice** priority is Augmentation Policy first, then additional losses plus scheduler or early stopping support, then Boundary Target, Temporal Input, and Approved Pretrained Encoders.
- The **Candidate Experiment Contract** never grants arbitrary filesystem access, network access, Docker access, dataset-path control, MLflow write access, custom training-loop authority, or custom data-loading authority.
- Candidate Experiments may reference **Approved Weight Artifacts** by stable identifier only.
- A **Pretrained Weight Request** may produce an **Approved Weight Artifact** after manual audit.
- Candidate Experiments must not download pretrained weights at runtime or reference arbitrary checkpoint paths.
- Candidate Experiment code runs without network access inside the **Candidate Execution Boundary**.
- Candidate Experiment code writes outputs only to the run-specific output directory; the **Harness** persists approved artifacts to MLflow.
- **Post-Run Evaluation** artifacts are stored under the original **Run** at `outputs/evaluations/` so their provenance remains attached to the completed **Run** they evaluate.
- The first **Post-Run Evaluation** mode is Whole-Validation Failure Analysis: it runs inference over the validation split, writes per-sample metrics for all evaluated samples, and writes bounded diagnostic artifacts for selected best, worst, false-positive-heavy, and false-negative-heavy cases by default.
- A **Post-Run Evaluation** uses the original **Run**'s **Resolved Manifest** as authoritative for model and data contract choices, while allowing Harness-owned diagnostic overrides such as thresholds, evaluation batch size, and artifact count.
- Whole-Validation Failure Analysis uses a default threshold sweep from `0.05` to `0.95` in `0.05` increments, while keeping `0.5` as the default binary-mask threshold.
- Whole-Validation Failure Analysis initially saves bounded diagnostic artifacts for `worst_by_dice`, `best_by_dice`, `false_positive_heavy`, `false_negative_heavy`, `empty_mask_false_positives`, and `missed_positive_masks` buckets.
- A **Post-Run Evaluation** has an Evaluation ID of the form `eval_YYYYMMDD_HHMMSS_<suffix>` and records its own lifecycle status (`running`, `completed`, or `failed`) without changing the parent **Run** status.
- The agent may have MLflow read access, explicitly configured read-only Research Problem data access, and normal pi-enclave network access constrained by Gondolin policy.
- The **Agent Control Boundary** constrains infrastructure authority rather than serving as the primary mechanism for validation/test overfitting control.
- The **Candidate Execution Boundary** constrains what Candidate Experiment code can do during a Run.
- The **Agent Workspace** is writable by the agent; the **Research History** is read-only prior research context.
- The **Agent Workspace** contains mutable draft Candidate Experiments under `drafts/candidates/`, immutable queued submissions under `submissions/`, and writable autonomous-loop artifacts such as Research Notes, Capability Requests, Evaluation Requests, Campaign Reports, and scratch files; it does not contain generic Research Ledger event requests.
- The **Agent Reference Snapshot** provides read-only setup-time copies of root canonical files such as `CONTEXT.md` and `EXPERIMENT_INDEX.md` because the Agent Control Boundary mounts directories rather than individual files.
- The **Research History** may use a generated parent directory for singleton files such as `research-ledger.jsonl` plus placeholder child directories that are over-mounted by read-only historical Candidate Experiment, Run, and Research Note directories.
- Canonical Research Ledger and Experiment Index updates are performed by Harness ingestion outside the **Agent Control Boundary**, not by direct agent edits.
- The **Candidate Submission Queue** separates draft **Candidate Experiments** from submissions that are ready for Harness validation and execution.
- Once placed in the **Candidate Submission Queue**, a **Candidate Experiment** is immutable by convention; changes require a new **Candidate Experiment** or a **Repair Candidate**.
- **Candidate Submission Preparation** happens in a minimal Agent Control Boundary Python environment without PyTorch or NVIDIA libraries and must not import Candidate Experiment model code.
- The Agent Control Boundary exposes an agent-safe CLI wrapper for observation and static Candidate Experiment submission preparation instead of exposing Run execution commands to the agent.
- The **Candidate Experiment Runner** bridges the **Agent Control Boundary** and the **Candidate Execution Boundary**.

## Example dialogue

> **Dev:** "Is this runner specific to contrail detection?"
> **Domain expert:** "No — **Ground-Camera Contrail Detection** is the first **Research Problem**, but the project is **ML Autoresearch**. The reusable execution subsystem is the **Candidate Experiment Runner**."
>
> **Dev:** "Should we call each submitted architecture a run?"
> **Domain expert:** "No. The submitted research idea is a **Candidate Experiment**. A **Run** is one execution attempt of that candidate, and the **Result** is what the run produces."
>
> **Dev:** "Is Docker the sandbox for the agent?"
> **Domain expert:** "No. The **Agent Control Boundary** limits the agent's direct access. Docker is part of the **Candidate Execution Boundary** for untrusted candidate code."
>
> **Dev:** "Are we building the whole runner before trying contrail models?"
> **Domain expert:** "No. The top-level plan is **Research Loop** first: define the Research Problem, run useful Candidate Experiments safely, learn from Results, and expand the system where the loop needs more freedom."
>
> **Dev:** "Can the agent write a custom training loop if it needs one?"
> **Domain expert:** "No. Training loops belong to the **Harness**. If a different loop is needed, expose it as an explicit Harness-owned parameter in the **Candidate Experiment Contract**."
>
> **Dev:** "Does the initial contract only allow model.py?"
> **Domain expert:** "No. The **Initial Flexibility Envelope** must be broad enough for effective research from the start, while still denying unsafe authority."
>
> **Dev:** "Can a Candidate Experiment include a URL to download pretrained weights?"
> **Domain expert:** "No. It can reference an **Approved Weight Artifact** by ID, or the agent can make a **Pretrained Weight Request** for manual audit."
>
> **Dev:** "Is the contrail problem object detection?"
> **Domain expert:** "No. **Ground-Camera Contrail Detection** predicts a **Contrail Mask**: binary semantic segmentation of contrail vs non-contrail pixels."
>
> **Dev:** "Does a temporal model predict a mask for the whole video clip?"
> **Domain expert:** "No. **Centered Temporal RGB Clip Input** provides temporal context around a **Target Frame**, and the model predicts the **Contrail Mask** for that Target Frame."
>
> **Dev:** "Are line logits a Hough transform prediction?"
> **Domain expert:** "No. The v1 **Line Target** is an image-aligned per-pixel **Auxiliary Target** derived from the **Contrail Mask** and used for an auxiliary loss."

## Flagged ambiguities

- "runner" was used to mean both the whole project and the trusted execution subsystem — resolved: the project is **ML Autoresearch**; the subsystem is the **Candidate Experiment Runner**.
- "candidate", "architecture", "package", and "experiment" were overlapping — resolved: the submitted research unit is a **Candidate Experiment**; the neural network design inside it is a **Model Architecture**.
- "safe" was used for both agent access control and untrusted code containment — resolved: use **Agent Control Boundary** for agent permissions and **Candidate Execution Boundary** for candidate-code execution.
- "task", "use case", and "test case" were used for the target ML problem — resolved: use **Research Problem**.
- "minimal contract" could mean minimal expressiveness or minimal implementation — resolved: the **Candidate Experiment Contract** minimizes unsafe authority while preserving enough research expressiveness through Harness-owned parameters.
- Wall-clock limits could prematurely bias architecture search or waste compute — resolved: keep **Wall-Clock Budget Policy** Harness-owned and explicitly adjustable, likely smaller during early exploration.
- "pretrained weights" could mean runtime downloads, arbitrary checkpoint paths, or audited reusable artifacts — resolved: Candidate Experiments may only reference **Approved Weight Artifacts**, and new weights enter through **Pretrained Weight Requests**.
- "detection" in **Ground-Camera Contrail Detection** could imply object detection — resolved: the prediction target is binary semantic segmentation producing a **Contrail Mask**.
- GVCCS whole-sky-camera data may not match likely downstream conventional ground-camera imagery — resolved: track this as **Camera Domain Shift**, but keep non-GVCCS evaluation outside the initial ML Autoresearch loop.
- "line logits" and "boundary logits" could imply non-image-space outputs — resolved: v1 auxiliary outputs are image-aligned per-pixel logits trained against Harness-derived **Auxiliary Targets**.
