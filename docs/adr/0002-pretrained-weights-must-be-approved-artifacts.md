# Pretrained weights must be approved artifacts

ML Autoresearch allows transfer learning, but Candidate Experiments must not download pretrained weights at runtime or reference arbitrary checkpoint paths. Candidate Experiments may reference only Approved Weight Artifacts by stable identifier; new weights enter the system through Pretrained Weight Requests that record source, license, intended use, and audit information before the Harness makes them available.
