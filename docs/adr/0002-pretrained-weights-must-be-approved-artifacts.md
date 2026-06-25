# Pretrained weights must be approved artifacts

ML Autoresearch allows transfer learning as future architecture, but current Candidate Experiments must not download pretrained weights at runtime, include checkpoint files, or reference arbitrary checkpoint paths.

The current implemented policy is negative enforcement: candidate source validation rejects checkpoint/weight artifacts and the Candidate Experiment Contract forbids runtime weight downloads. Approved Weight Artifacts by stable identifier and Pretrained Weight Requests that record source, license, intended use, and audit information remain follow-on design intent until a registry/workflow is implemented and tested.
