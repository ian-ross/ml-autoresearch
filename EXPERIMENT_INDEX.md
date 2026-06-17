# Experiment Index

This index links Candidate Experiment descriptions to the Research Notes that interpret their Runs and Post-Run Evaluations.

**Maintenance rule:** update this file whenever a new Candidate Experiment is introduced or a new Research Note is written.

## Current baseline

- Current in-contract GVCCS baseline: [`candidates/single_frame_small_unet_realistic_training_shuffled`](candidates/single_frame_small_unet_realistic_training_shuffled/README.md)
- Most recent Research Note: [`research-notes/2026-05-08-small-unet-line-auxiliary.md`](research-notes/2026-05-08-small-unet-line-auxiliary.md)
- Baseline Run: `run_20260506_045020_84aac5`
- Latest baseline Post-Run Evaluation: `eval_20260507_104724_a96db5`
- Current challenger Run: `run_20260507_193004_0b4688`

## Candidate Experiments and notes

| Candidate Experiment | Description | Related Research Notes | Key Runs / Evaluations | Status |
| --- | --- | --- | --- | --- |
| `tests/fixtures/candidates/single_frame_unet_baseline` | Harness fixture candidate; no standalone README. | [`2026-05-04 baseline GVCCS tiny subset`](research-notes/2026-05-04-baseline-gvccs-tiny-subset.md); [`2026-05-04 small U-Net tiny subset comparison`](research-notes/2026-05-04-small-unet-tiny-subset-comparison.md) | `run_20260504_192239_b2ebd1` | Harness/debug baseline only; not a model-quality baseline. |
| `candidates/single_frame_small_unet` | [`README.md`](candidates/single_frame_small_unet/README.md) | [`2026-05-04 small U-Net tiny subset comparison`](research-notes/2026-05-04-small-unet-tiny-subset-comparison.md) | `run_20260504_202102_9b2093` | Tiny-subset architecture check; inconclusive for model quality. |
| `candidates/single_frame_small_unet_realistic_training` | [`README.md`](candidates/single_frame_small_unet_realistic_training/README.md) | [`2026-05-05 small U-Net realistic training`](research-notes/2026-05-05-small-unet-realistic-training.md) | `run_20260505_092224_fa11f8` | Weak realistic baseline; superseded by shuffled training. |
| `candidates/single_frame_small_unet_realistic_training_shuffled` | [`README.md`](candidates/single_frame_small_unet_realistic_training_shuffled/README.md) | [`2026-05-06 small U-Net shuffled realistic training`](research-notes/2026-05-06-small-unet-shuffled-realistic-training.md); [`2026-05-07 small U-Net post-run evaluation`](research-notes/2026-05-07-small-unet-post-run-evaluation.md) | `run_20260505_172954_febf68`; `run_20260506_045020_84aac5`; `eval_20260507_104724_a96db5` | Current in-contract GVCCS baseline. |
| `candidates/single_frame_small_unet_line_auxiliary` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary/README.md) | [`2026-05-08 small U-Net line auxiliary target`](research-notes/2026-05-08-small-unet-line-auxiliary.md) | `run_20260507_193004_0b4688`; `eval_20260508_100735_6fea74`; compared to `run_20260506_045020_84aac5` / `eval_20260507_104724_a96db5` | Promising challenger; modest Dice/recall gain, tune auxiliary weight next. |
| `candidates/single_frame_small_unet_line_auxiliary_w010` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary_w010/README.md) | Pending full GVCCS Research Note. | Pending GVCCS Run; compare to `run_20260506_045020_84aac5` and `run_20260507_193004_0b4688`. | Introduced; tests lower Line Target auxiliary weight `0.10`. |

| `candidates/single_frame_small_unet_line_auxiliary_w010_light_combined` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary_w010_light_combined/README.md) — Small U-Net single-frame segmentation candidate with line auxiliary target weight 0.10 and Harness-owned light combined augmentation. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_small_unet_line_auxiliary_w010_light_geometric` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary_w010_light_geometric/README.md) — Small U-Net single-frame segmentation candidate with line auxiliary target weight 0.10 and Harness-owned light geometric augmentation. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_small_unet_line_auxiliary_w010_light_photometric` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary_w010_light_photometric/README.md) — Small U-Net single-frame segmentation candidate with line auxiliary target weight 0.10 and Harness-owned light photometric augmentation. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_medium_unet_line_auxiliary_w010` | [`README.md`](candidates/single_frame_medium_unet_line_auxiliary_w010/README.md) — Medium-capacity single-frame U-Net with Harness-derived line auxiliary target at weight 0.10. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_medium_unet_line_auxiliary_w005` | [`README.md`](candidates/single_frame_medium_unet_line_auxiliary_w005/README.md) — Medium-capacity single-frame U-Net with Harness-derived line auxiliary target at weight 0.05 for a recall-oriented follow-up. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_medium_unet_line_auxiliary_w010_light_photometric` | [`README.md`](candidates/single_frame_medium_unet_line_auxiliary_w010_light_photometric/README.md) — Medium-capacity single-frame U-Net with Harness-derived line auxiliary target at weight 0.10 and light photometric augmentation. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_large_unet_line_auxiliary_w010` | [`README.md`](candidates/single_frame_large_unet_line_auxiliary_w010/README.md) — Larger single-frame U-Net with Harness-derived line auxiliary target at weight 0.10. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_wide_unet_line_auxiliary_w010` | [`README.md`](candidates/single_frame_wide_unet_line_auxiliary_w010/README.md) — Wider single-frame U-Net with Harness-derived line auxiliary target at weight 0.10. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_auxiliary_w010` | [`README.md`](candidates/single_frame_xwide_unet_line_auxiliary_w010/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target at weight 0.10. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target and bottleneck dropout regularization. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_attention_dropout` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_attention_dropout/README.md) — Extra-wide single-frame U-Net with line auxiliary target, bottleneck dropout, and lightweight decoder attention gates on skip features. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p015` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p015/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target and slightly stronger bottleneck dropout regularization. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target and lighter bottleneck dropout regularization. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target and lighter bottleneck dropout regularization; resubmission after a non-scientific Harness interruption. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075` | [`README.md`](candidates/single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075/README.md) — Extra-wide single-frame U-Net with p=0.075 bottleneck dropout plus Harness-derived line and boundary auxiliary targets. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine/README.md) — Extra-wide single-frame U-Net with line auxiliary target, light dropout, and a high-resolution residual refinement block for tiny contrail masks. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target, lighter bottleneck dropout regularization, and a 40-epoch training budget to test whether the current best was still improving at epoch 30. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40` | [`README.md`](candidates/single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40/README.md) — Extra-wide single-frame U-Net with p=0.075 bottleneck dropout, Harness-derived line auxiliary target, conservative boundary auxiliary target, and a 40-epoch training budget. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target, very light bottleneck dropout regularization and a 40-epoch training budget to test whether p=0.05 improves recall without losing the current best precision control. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40` | [`README.md`](candidates/single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40/README.md) — Extra-wide single-frame U-Net with p=0.075 bottleneck dropout, Harness-derived line auxiliary target, very-low-weight boundary auxiliary target, and a 40-epoch training budget. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target, p=0.075 bottleneck dropout, and a 60-epoch training budget to test whether the recall-safer 40-epoch baseline continues improving. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse/README.md) — Extra-wide single-frame U-Net with Harness-derived line auxiliary target, p=0.075 bottleneck dropout, 60 epochs, and a shallow high-resolution detail-fusion head to test whether preserving encoder detail reduces tiny-mask misses and large-mask under-segmentation. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005/README.md) — Extra-wide single-frame U-Net with line auxiliary target, p=0.075 bottleneck dropout, and light p=0.05 decoder-head dropout to test precision-safe regularization at the 60-epoch budget. | Pending full GVCCS Research Note. | Pending GVCCS Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es/README.md) — Current-best extra-wide single-frame U-Net with Line Target auxiliary loss, p=0.075 bottleneck dropout, Harness-owned reduce-on-plateau scheduling, and early stopping with best-checkpoint restoration. | Pending full Research Note. | Pending Research Problem Run. | Pending Harness Run; ingested from Agent Workspace. |

| `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context` | [`README.md`](candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context/README.md) — Extra-wide single-frame U-Net with Line Target auxiliary loss, p=0.075 bottleneck dropout, lightweight dilated bottleneck context, reduce-on-plateau scheduling, and early stopping with best-checkpoint restoration. | Pending full Research Note. | Pending Research Problem Run. | Pending Harness Run; ingested from Agent Workspace. |

## Chronological Research Notes

- [`2026-05-04 baseline GVCCS tiny subset`](research-notes/2026-05-04-baseline-gvccs-tiny-subset.md)
- [`2026-05-04 small U-Net tiny subset comparison`](research-notes/2026-05-04-small-unet-tiny-subset-comparison.md)
- [`2026-05-05 small U-Net realistic training`](research-notes/2026-05-05-small-unet-realistic-training.md)
- [`2026-05-06 small U-Net shuffled realistic training`](research-notes/2026-05-06-small-unet-shuffled-realistic-training.md)
- [`2026-05-07 small U-Net post-run evaluation`](research-notes/2026-05-07-small-unet-post-run-evaluation.md)
- [`2026-05-08 small U-Net line auxiliary target`](research-notes/2026-05-08-small-unet-line-auxiliary.md)
- [`Research Note: Lower-weight Line Target auxiliary Small U-Net`](research-notes/2026-05-22-small-unet-line-auxiliary-w010.md)
- [`2026-05-23 lower-weight Line Target auxiliary evaluation follow-up`](research-notes/2026-05-23-line-auxiliary-w010-evaluation.md)
- [`2026-05-25 light-combined augmentation regression`](research-notes/2026-05-25-light-combined-augmentation-regression.md)
- [`2026-05-25 light-geometric augmentation regression`](research-notes/2026-05-25-light-geometric-augmentation-regression.md)
- [`2026-05-29 light-photometric augmentation near miss`](research-notes/2026-05-29-light-photometric-augmentation-near-miss.md)
- [`2026-05-29 medium U-Net Line Target auxiliary improvement`](research-notes/2026-05-29-medium-unet-line-auxiliary-w010-improvement.md)
- [`2026-05-30 medium U-Net Line Target auxiliary w0.05 regression`](research-notes/2026-05-30-medium-unet-line-auxiliary-w005-regression.md)
- [`2026-05-30 medium U-Net light photometric augmentation regression`](research-notes/2026-05-30-medium-unet-light-photometric-regression.md)
- [`2026-05-30 large U-Net Line Target auxiliary improvement`](research-notes/2026-05-30-large-unet-line-auxiliary-w010-improvement.md)
- [`2026-05-30 wide U-Net Line Target auxiliary improvement`](research-notes/2026-05-30-wide-unet-line-auxiliary-w010-improvement.md)
- [`2026-05-30 wide U-Net failure-bucket evaluation`](research-notes/2026-05-30-wide-unet-failure-bucket-evaluation.md)
- [`2026-05-30 extra-wide U-Net Line Target auxiliary improvement`](research-notes/2026-05-30-xwide-unet-line-auxiliary-w010-improvement.md)
- [`2026-05-30 xwide U-Net failure-bucket evaluation`](research-notes/2026-05-30-xwide-unet-failure-bucket-evaluation.md)
- [`2026-05-31 xwide U-Net bottleneck dropout improvement`](research-notes/2026-05-31-xwide-unet-dropout-improvement.md)
- [`2026-05-31 xwide dropout failure-bucket evaluation`](research-notes/2026-05-31-xwide-dropout-failure-bucket-evaluation.md)
- [`2026-05-31 xwide attention-dropout regression`](research-notes/2026-05-31-xwide-attention-dropout-regression.md)
- [`2026-05-31 xwide U-Net stronger bottleneck dropout regression`](research-notes/2026-05-31-xwide-dropout-p015-regression.md)
- [`2026-06-01 xwide U-Net lighter bottleneck dropout improvement`](research-notes/2026-06-01-xwide-dropout-p0075-improvement.md)
- [`2026-06-01 xwide p=0.075 dropout failure-bucket evaluation`](research-notes/2026-06-01-xwide-dropout-p0075-failure-bucket-evaluation.md)
- [`2026-06-01 boundary auxiliary training contract failure`](research-notes/2026-06-01-boundary-auxiliary-training-contract-failure.md)
- [`2026-06-01 high-resolution refinement regression`](research-notes/2026-06-01-xwide-refinement-regression.md)
- [`2026-06-01 xwide p=0.075 dropout 40-epoch improvement`](research-notes/2026-06-01-xwide-dropout-p0075-epoch40-improvement.md)
- [`2026-06-01 xwide p=0.075 dropout 40-epoch failure-bucket evaluation`](research-notes/2026-06-01-xwide-dropout-p0075-epoch40-failure-bucket-evaluation.md)
- [`2026-06-02 boundary auxiliary w=0.03 epoch-40 regression`](research-notes/2026-06-02-boundary-auxiliary-w003-epoch40-regression.md)
- [`2026-06-02 xwide p=0.05 dropout 40-epoch precision-biased improvement`](research-notes/2026-06-02-xwide-dropout-p005-epoch40-precision-biased-improvement.md)
- [`2026-06-02 xwide p=0.05 dropout 40-epoch failure-bucket evaluation`](research-notes/2026-06-02-xwide-dropout-p005-epoch40-failure-bucket-evaluation.md)
- [`2026-06-02 boundary auxiliary w=0.01 epoch-40 regression`](research-notes/2026-06-02-boundary-auxiliary-w001-epoch40-regression.md)
- [`2026-06-03 xwide p=0.075 dropout 60-epoch improvement`](research-notes/2026-06-03-xwide-dropout-p0075-epoch60-improvement.md)
- [`2026-06-03 xwide p=0.075 dropout 60-epoch failure-bucket evaluation`](research-notes/2026-06-03-xwide-dropout-p0075-epoch60-failure-bucket-evaluation.md)
- [`2026-06-03 detail-fusion architecture regression`](research-notes/2026-06-03-detail-fusion-regression.md)
- [`2026-06-15 reduce-on-plateau early-stopping training-policy improvement`](research-notes/2026-06-15-plateau-es-training-policy-improvement.md)
- [`2026-06-15 plateau/es failure-bucket evaluation`](research-notes/2026-06-15-plateau-es-failure-bucket-evaluation.md)
