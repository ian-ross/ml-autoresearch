# ML Autoresearch

ML Autoresearch is a safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments. The initial Research Problem is Ground-Camera Contrail Detection, but the system is intended to support other Research Problems later.

## Language

**ML Autoresearch**:
A safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments.
_Avoid_: Contrail runner, Docker runner, Pi runner

**Research Problem**:
A target ML problem that ML Autoresearch explores through Candidate Experiments, including its data modality, prediction target, evaluation metrics, and constraints.
_Avoid_: Task, use case, project

**Research Loop**:
The closed cycle in which the agent proposes Candidate Experiments, the system runs them, Results are evaluated, and the next Candidate Experiment is informed by prior Results.
_Avoid_: Pipeline when referring to the iterative research cycle

**Candidate Experiment**:
An agent-proposed, runnable ML research package containing the model architecture and allowed configuration needed for the runner to evaluate one research idea.
_Avoid_: Candidate, architecture, package, submission when referring to the whole unit

**Candidate Experiment Contract**:
The allowlisted interface through which a Candidate Experiment can express research variation without receiving unsafe authority over training, data loading, filesystem access, network access, Docker, or MLflow writes.
_Avoid_: Free-form plugin API, arbitrary experiment code

**Initial Flexibility Envelope**:
The v1 set of research variations exposed by the Candidate Experiment Contract: model architecture, input mode, output form, loss selection, bounded training knobs, augmentation/data policy, and pretrained weight requests.
_Avoid_: Minimal contract when that implies low research expressiveness

**Wall-Clock Budget Policy**:
The Harness-owned policy limiting Run duration, expected to be adjustable so early research can prefer many cheap experiments before longer training runs.
_Avoid_: Fixed training duration baked into Candidate Experiments

**Resolved Manifest**:
The Harness-owned record of the Candidate Experiment manifest after validation, normalization, and Harness defaults have been applied for a specific Run.
_Avoid_: Source manifest, candidate configuration

**Harness**:
The trusted outer implementation that owns training loops, data loading, validation, execution policy, artifact persistence, and approved parameterized variations.
_Avoid_: Candidate code, agent code

**Model Architecture**:
The neural network design inside a Candidate Experiment.
_Avoid_: Candidate experiment when referring only to the network design

**Run**:
One execution attempt of a Candidate Experiment by the Candidate Experiment Runner.
_Avoid_: Experiment when referring only to execution

**Result**:
The metrics and artifacts produced by a Run.
_Avoid_: Run when referring only to observed outputs

**Candidate Experiment Runner**:
The trusted subsystem that validates, executes, and records agent-proposed Candidate Experiments without exposing host secrets, datasets, Docker, or GPU control to the agent.
_Avoid_: Pi runner, Docker runner, contrail runner

**Agent Control Boundary**:
The boundary that prevents the agent from directly accessing host shell, Docker, datasets, secrets, cloud credentials, or unrestricted network resources.
_Avoid_: Sandbox when referring specifically to agent permissions

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
The initial Research Problem for ML Autoresearch, focused on binary semantic segmentation of contrail pixels vs non-contrail pixels in ground-camera imagery.
_Avoid_: The project, the product

**GVCCS Dataset**:
The whole-sky-camera training dataset for the Ground-Camera Contrail Detection Research Problem, published at https://zenodo.org/records/16612390.
_Avoid_: Contrail dataset when a specific dataset identity is needed

**Camera Domain Shift**:
The expected distribution difference between GVCCS whole-sky-camera training data and likely downstream conventional ground-camera imagery.
_Avoid_: Treating GVCCS validation as deployment validation

**Contrail Mask**:
A binary semantic segmentation label marking each pixel as contrail or non-contrail.
_Avoid_: Classification label, detection box

**Auxiliary Target**:
A Harness-derived per-pixel training target used for an auxiliary loss, not a separate primary prediction target.
_Avoid_: Primary label, end-user prediction

**Line Target**:
An Auxiliary Target derived from the Contrail Mask to emphasize thin centerline-like contrail structure.
_Avoid_: Hough-space target, line detection output

**Boundary Target**:
An Auxiliary Target derived from the Contrail Mask to emphasize contrail edge geometry.
_Avoid_: Object-detection boundary, separate segmentation class

**Target Frame**:
The frame whose Contrail Mask a Candidate Experiment predicts.
_Avoid_: Label frame, center image

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
- **Ground-Camera Contrail Detection** uses the **GVCCS Dataset** for training.
- **Camera Domain Shift** is a known limitation of using the **GVCCS Dataset** for models likely to be tried on conventional ground-camera imagery.
- Evaluation on non-GVCCS camera data is a separate exercise outside the initial **ML Autoresearch** loop.
- The prediction target for **Ground-Camera Contrail Detection** is a **Contrail Mask** for a **Target Frame**.
- **Line Target** and **Boundary Target** are optional **Auxiliary Targets** derived from the **Contrail Mask** by the **Harness**.
- **Auxiliary Targets** are used for auxiliary training losses; primary evaluation remains based on the **Contrail Mask** prediction.
- **Single-Frame RGB Input** and **Centered Temporal RGB Clip Input** are v1 **Input Modes** for **Ground-Camera Contrail Detection**.
- **ML Autoresearch** works by sustaining a **Research Loop**.
- A **Research Loop** explores a **Research Problem** through many **Candidate Experiments**.
- **ML Autoresearch** uses a **Candidate Experiment Runner** to execute agent-proposed **Candidate Experiments** safely.
- A **Research Problem** is explored through many **Candidate Experiments**.
- A **Candidate Experiment** conforms to a **Candidate Experiment Contract**.
- A **Candidate Experiment** may have a **Resolved Manifest** for each accepted **Run**.
- A **Resolved Manifest** records what the **Harness** actually ran, not just what the Candidate Experiment requested.
- A **Candidate Experiment** contains one **Model Architecture**.
- A **Candidate Experiment** may produce zero or more **Runs**.
- A **Run** produces at most one **Result**.
- The **Initial Flexibility Envelope** is part of the **Candidate Experiment Contract** from the beginning.
- The **Initial Flexibility Envelope** includes model architecture, input mode, output form, loss selection, bounded training knobs, augmentation/data policy, and pretrained weight requests.
- **Wall-Clock Budget Policy** is intentionally adjustable and may start small to encourage many cheap architecture-search Runs.
- The **Harness** owns training loops, data loading, validation, execution policy, artifact persistence, and approved parameterized variations.
- The **Candidate Experiment Contract** can expose Harness-owned parameters for model architecture, input modes, output forms, losses, optimizer choices, training budgets, augmentation/data policy, and pretrained weight availability.
- The **Candidate Experiment Contract** never grants arbitrary filesystem access, network access, Docker access, dataset-path control, MLflow write access, custom training-loop authority, or custom data-loading authority.
- Candidate Experiments may reference **Approved Weight Artifacts** by stable identifier only.
- A **Pretrained Weight Request** may produce an **Approved Weight Artifact** after manual audit.
- Candidate Experiments must not download pretrained weights at runtime or reference arbitrary checkpoint paths.
- Candidate Experiment code runs without network access inside the **Candidate Execution Boundary**.
- Candidate Experiment code writes outputs only to the run-specific output directory; the **Harness** persists approved artifacts to MLflow.
- The agent may have MLflow read access and normal pi-enclave network access constrained by Gondolin policy.
- The **Agent Control Boundary** constrains what the agent can directly access.
- The **Candidate Execution Boundary** constrains what Candidate Experiment code can do during a Run.
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
