# 2026-05-04 baseline GVCCS tiny subset

## Hypothesis

The primary hypothesis for this first Human-Guided Research Iteration was operational rather than architectural: the current Harness should be able to run a Candidate Experiment through Docker-backed `run-candidate` on a tiny real GVCCS subset and produce enough Result artifacts to support a next research decision.

A secondary, deliberately weak baseline expectation was that the fixture model would not be competitive for Ground-Camera Contrail Detection. It has only 233 trainable parameters and was originally intended to validate the Harness path, not to serve as a realistic segmentation architecture.

## Candidate Experiment(s)

- Candidate Experiment path: `tests/fixtures/candidates/single_frame_unet_baseline`
- Relevant Experiment Proposal: none; this is the initial post-tracer-bullet baseline Run for Human-Guided Research Iteration 1.

## Run(s)

- Run ID: `run_20260504_192239_b2ebd1`
- Run directory: `runs/run_20260504_192239_b2ebd1`
- Dataset mode/subset: GVCCS Dataset via `--data-root`; tiny subset with `--max-samples 8`, producing 6 train samples and 2 validation samples.
- Harness/backend notes: Docker backend using `ml-autoresearch-runner:local`; rootless Docker ownership mode recorded in metadata (`docker_user: 0:0`, `user_namespace: rootless`); GPU disabled by default; resource limits recorded as 4 GiB memory, 2 CPUs, 512 pids, 2 GiB scratch.
- Training configuration from Resolved Manifest: single-frame RGB input, mask-logits output, `bce_dice`, AdamW, learning rate 0.001, batch size 2, 1 epoch.

## Key metrics

| Run | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260504_192239_b2ebd1` | 0.0092345 | 0.0046387 | 0.0046387 | 1.0 | 1.7275207 |

Additional artifact notes:

- `outputs/model_summary.json` reports only 233 parameters.
- `outputs/metrics.jsonl` contains three training batches and one validation summary.
- `outputs/prediction_samples/samples.json` contains two validation samples, each with input, ground-truth, prediction, and overlay images.

## Qualitative observations

The Result is useful as a Harness/debugging baseline, but not as evidence for a viable Model Architecture.

The tiny model appears to predict the positive class too broadly: validation recall is 1.0 while precision is approximately the positive-pixel fraction, giving near-zero Dice/IoU. With only 233 parameters and one epoch on 6 training / 2 validation samples, this is expected and should not be over-interpreted as a meaningful architecture comparison.

The important qualitative observation is that the artifact set is complete enough for human inspection and next-step selection:

- metrics were written in both streaming and final forms;
- model summary exposed the architecture scale problem immediately;
- prediction sample metadata and image artifacts were generated;
- run metadata recorded the dataset mount, backend, and execution policy.

## Decision

Do not continue with this fixture model as a research baseline. Keep it as a Harness smoke/debug Candidate Experiment.

The Run verifies that the Human-Guided Research Iteration path is viable: `run-candidate` can run on a tiny GVCCS subset, produce metrics, logs, summaries, and prediction samples, and preserve enough context for a Research Note.

The current artifacts were sufficient to choose a next Candidate Experiment. In fact, the model summary alone was enough to show that the current Candidate Experiment is too small to test the Research Problem seriously; the metrics and prediction samples confirm that it is not learning a useful mask under this tiny setup.

## Next proposed change

Run a more realistic in-contract single-frame segmentation architecture next, while keeping the Harness contract unchanged:

- Single-Frame RGB Input;
- mask-logits output only;
- `bce_dice` loss;
- AdamW optimizer;
- no temporal inputs, auxiliary heads, pretrained weights, MLflow, async orchestration, or broader contract changes yet.

The next Candidate Experiment should be a standard small U-Net-style encoder/decoder with skip connections and enough capacity to make the tiny GVCCS subset a useful end-to-end check. The goal is not to claim real validation performance from 8 samples, but to establish a credible baseline architecture and verify that the artifacts remain informative when the model has enough capacity to plausibly fit simple contrail masks.
