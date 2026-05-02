# Pi Enclave + Docker + Trusted Host Runner for ML Architecture Search

> Historical design brief. `CONTEXT.md`, `docs/top-level-plan.md`, `docs/tracer-bullet-issues.md`, and the current GitHub issues are canonical for implementation sequencing. Where this document differs from those sources or the current code, treat it as background rationale rather than the active contract.

This document describes a practical setup for running the Pi coding agent inside `pi-enclave`/Gondolin while delegating GPU training to a trusted host-side runner that uses Docker.

The intended use case is agent-assisted ML architecture search for contrail detection in video from ground-based cameras, with a path toward more complex satellite/multispectral experiments later.

The key design goal is:

> Let the agent propose and iterate on ML models, but do not give the agent direct shell access to the GPU host, Docker daemon, dataset directories, secrets, or arbitrary network resources.

---

## 1. Architecture overview

```text
+-----------------------------+
| Pi agent                    |
| running inside pi-enclave   |
| /workspace mounted          |
+--------------+--------------+
               |
               | writes candidate code/config
               | calls narrow host-facing commands
               v
+-----------------------------+
| Trusted host runner         |
| outside Gondolin            |
| validates candidate package |
| launches Docker GPU worker  |
| owns MLflow write access     |
+--------------+--------------+
               |
               | docker run --gpus ...
               v
+-----------------------------+
| Docker training container   |
| imports candidate model     |
| trains/evaluates on GPU     |
| writes outputs/artifacts    |
+--------------+--------------+
               |
               v
+-----------------------------+
| MLflow / output store       |
| metrics, params, artifacts  |
+-----------------------------+
```

The agent should not be able to do any of the following directly:

- SSH to the GPU node.
- Invoke `docker`.
- Access `/var/run/docker.sock`.
- Read arbitrary home directories.
- Read credentials or cloud tokens.
- Write outside the candidate/output areas.
- Upload artifacts directly to MLflow using a write/admin token.

The trusted runner is the only component allowed to launch Docker GPU containers or write official experiment results to MLflow.

---

## 2. Threat model

This setup is not meant to prove that arbitrary generated Python code is safe. It assumes candidate code may be buggy, surprising, or even actively malicious.

The security boundary is therefore:

```text
trusted host runner + Docker isolation + strict mounts + scoped secrets
```

not:

```text
Python audit hooks
static code review
LLM review
```

Audit checks and second-agent review are useful, but they are defense-in-depth. They are not a sandbox.

---

## 3. Directory layout

A suggested layout on the GPU node:

```text
/opt/contrail-agent-runner/
  runner/
    submit_candidate.py
    worker_entrypoint.py
    validate_candidate.py
    docker_run.py
    mlflow_upload.py
  images/
    Dockerfile
  schemas/
    candidate.schema.json

/srv/contrail-agent/
  candidates/
    pending/
    accepted/
    rejected/
  runs/
    run-000001/
      candidate/
      outputs/
      logs/
      metadata.json
  datasets/
    ground-camera-v1/       # read-only to containers
  secrets/
    mlflow.env              # readable only by trusted runner user
```

A candidate package produced by the agent might live inside the project workspace:

```text
candidate/
  manifest.yaml
  model.py
  transforms.py             # optional
  loss.py                   # optional, maybe disallowed initially
  README.md
```

The trusted runner copies the candidate into a run-specific directory before executing it:

```text
/srv/contrail-agent/runs/run-000042/candidate/
/srv/contrail-agent/runs/run-000042/outputs/
```

---

## 4. pi-enclave role

`pi-enclave`/Gondolin should be used to isolate the agent control plane.

Recommended pi-enclave posture:

```text
workspace mount:
  repo: read/write

extra mounts:
  none by default

network:
  default deny
  optionally allow:
    - internal Git server
    - package registry/mirror if needed
    - MLflow read-only endpoint, if safe

secrets:
  no write/admin MLflow token
  no SSH private keys
  no Docker socket
  no cloud credentials
```

The agent should be given access only to wrapper commands such as:

```bash
submit_candidate ./candidate
list_runs
get_run_summary run-000042
get_best_runs --metric val/iou --limit 20
```

These commands should call trusted host-side scripts. They should not be arbitrary shell escapes.

---

## 5. Docker role

Docker is used only by the trusted host runner, not by the agent.

A representative invocation:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=8g \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 4096 \
  --memory 96g \
  --shm-size 16g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$RUN_DIR/candidate:/candidate:ro" \
  -v "$DATASET_DIR:/data:ro" \
  -v "$RUN_DIR/outputs:/outputs:rw" \
  contrail-training:latest \
  python /runner/worker_entrypoint.py \
    --manifest /candidate/manifest.yaml \
    --output-dir /outputs
```

Important rules:

1. Never mount `/var/run/docker.sock` into the agent environment or candidate container.
2. Never mount the user's home directory into the training container.
3. Mount candidate code read-only.
4. Mount datasets read-only.
5. Mount only a run-specific output directory read/write.
6. Prefer no network inside the container.
7. Let the trusted runner upload results to MLflow after the container exits.

If the container must write directly to MLflow, inject only a scoped write token and restrict network egress as much as practical. However, the safer pattern is:

```text
container writes metrics/artifacts locally
trusted runner uploads to MLflow
```

---

## 6. Trusted host runner responsibilities

The trusted runner owns:

- Candidate validation.
- Run directory creation.
- Docker invocation.
- Resource limits.
- Dataset path selection.
- MLflow write token.
- Metric/artifact upload.
- Cleanup.
- Rejection of suspicious candidates.

The runner should reject candidates that:

- Use absolute paths.
- Attempt to import networking libraries.
- Attempt to import `os`, `subprocess`, `socket`, `ctypes`, `pickle`, `dill`, `cloudpickle`, `importlib`, or similar dangerous modules.
- Define custom training loops unless explicitly permitted.
- Read environment variables except allowlisted ones.
- Write outside `/outputs`.
- Try to use pretrained weights or download weights.
- Try to load arbitrary serialized Python objects.
- Use `torch.load` on agent-supplied files.
- Create symlinks.
- Spawn uncontrolled subprocesses.
- Attempt multiprocessing beyond an allowed dataloader configuration.

Static checks are not enough, but they are valuable for rejecting obviously bad candidates before Docker execution.

---

## 7. Candidate experiment contract

The candidate contract should be narrow at the system boundary, but broad enough to allow meaningful model architecture search.

A good initial contract is:

```text
The agent may provide:
  - manifest.yaml
  - model.py
  - optional transforms.py
  - optional loss.py, if enabled

The agent may not provide:
  - a full training loop
  - arbitrary shell scripts
  - Dockerfiles
  - data loading code with arbitrary paths
  - MLflow logging code
  - network access code
```

The trusted runner provides:

```python
input_spec = {
    "channels": 3,
    "height": 512,
    "width": 512,
    "sequence_length": 1,
}

output_spec = {
    "task": "binary_segmentation",
    "classes": 1,
}
```

The candidate must expose:

```python
def build_model(input_spec: dict, output_spec: dict) -> torch.nn.Module:
    ...
```

The model must accept tensors of shape:

```text
[B, C, H, W]
```

or, for temporal models:

```text
[B, T, C, H, W]
```

and return logits:

```text
[B, 1, H, W]
```

for binary segmentation.

---

## 8. Example architecture 1: lightweight UNet for RGB frame segmentation

This is a sensible first baseline for contrails in ordinary ground-camera video. Contrails are thin, elongated structures, so preserving spatial detail matters.

### YAML representation

```yaml
name: unet_rgb_baseline_v1
task: binary_segmentation

input:
  modality: ground_camera_rgb
  tensor_shape: [batch, 3, 512, 512]
  sequence_length: 1

model:
  family: unet
  entrypoint: model:build_model
  params:
    in_channels: 3
    out_channels: 1
    base_channels: 32
    depth: 4
    norm: batchnorm
    activation: relu
    upsampling: bilinear
    skip_connections: true

training:
  loss:
    type: bce_dice
    bce_weight: 0.5
    dice_weight: 0.5
  optimizer:
    type: adamw
    learning_rate: 0.0003
    weight_decay: 0.01
  batch_size: 16
  max_epochs: 40

constraints:
  allow_pretrained: false
  max_parameters: 20000000
  max_train_hours: 8
```

### PyTorch expansion

```python
# candidate/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="batchnorm"):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
        if norm == "batchnorm":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "groupnorm":
            layers.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))
        layers += [nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
        if norm == "batchnorm":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "groupnorm":
            layers.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, depth=4):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.down_blocks.append(ConvBlock(prev, ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            prev = ch
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev = channels[-1] * 2
        for ch in reversed(channels):
            self.up_convs.append(nn.Conv2d(prev, ch, kernel_size=1))
            self.up_blocks.append(ConvBlock(ch * 2, ch))
            prev = ch
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up_conv, up_block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        return self.head(x)


def build_model(input_spec, output_spec):
    return LightweightUNet(input_spec["channels"], output_spec["classes"], base_channels=32, depth=4)
```

---

## 9. Example architecture 2: temporal UNet with stacked frames

Contrails in ground-camera video often become easier to detect when temporal persistence and motion are available. A low-friction temporal model is to stack nearby frames as channels, then use a normal 2D segmentation network.

### YAML representation

```yaml
name: temporal_stack_unet_v1
task: binary_segmentation

input:
  modality: ground_camera_rgb_video
  tensor_shape: [batch, time, 3, 512, 512]
  sequence_length: 5
  temporal_encoding:
    mode: stack_rgb_frames
    include_center_frame: true

model:
  family: temporal_channel_stack_unet
  entrypoint: model:build_model
  params:
    effective_in_channels: 15
    out_channels: 1
    base_channels: 48
    depth: 4
    norm: groupnorm
    upsampling: bilinear

training:
  loss:
    type: focal_dice
    focal_gamma: 2.0
    focal_weight: 0.5
    dice_weight: 0.5
  optimizer:
    type: adamw
    learning_rate: 0.0002
    weight_decay: 0.02
  batch_size: 8
  max_epochs: 50

constraints:
  allow_pretrained: false
  max_parameters: 40000000
  max_train_hours: 12
```

### PyTorch expansion

```python
# candidate/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalStackUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, base_channels=48):
        super().__init__()
        self.enc1 = ConvGNReLU(in_channels, base_channels)
        self.enc2 = ConvGNReLU(base_channels, base_channels * 2)
        self.enc3 = ConvGNReLU(base_channels * 2, base_channels * 4)
        self.enc4 = ConvGNReLU(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvGNReLU(base_channels * 8, base_channels * 16)
        self.dec4 = ConvGNReLU(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = ConvGNReLU(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvGNReLU(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvGNReLU(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _stack_temporal_channels(self, x):
        b, t, c, h, w = x.shape
        return x.reshape(b, t * c, h, w)

    def forward(self, x):
        if x.ndim == 5:
            x = self._stack_temporal_channels(x)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def build_model(input_spec, output_spec):
    sequence_length = input_spec.get("sequence_length", 5)
    effective_in_channels = sequence_length * input_spec["channels"]
    return TemporalStackUNet(effective_in_channels, output_spec["classes"], base_channels=48)
```

---

## 10. Example architecture 3: UNet with line-enhancement auxiliary head

Contrails are long, narrow, high-aspect-ratio features. A useful family of models is a segmentation model with an auxiliary head that encourages the network to learn line-like structures.

The trusted runner can decide whether to use the auxiliary output. The candidate model returns either:

```python
{"mask_logits": ..., "line_logits": ...}
```

or just a tensor. The runner should explicitly permit dictionary outputs before enabling this architecture family.

### YAML representation

```yaml
name: line_aware_unet_v1
task: binary_segmentation

input:
  modality: ground_camera_rgb
  tensor_shape: [batch, 3, 512, 512]
  sequence_length: 1

model:
  family: line_aware_unet
  entrypoint: model:build_model
  params:
    in_channels: 3
    out_channels: 1
    base_channels: 32
    auxiliary_heads:
      line_head: true
      orientation_head: false

training:
  output_contract: dict
  losses:
    mask_logits:
      type: bce_dice
      weight: 1.0
    line_logits:
      type: bce
      weight: 0.25
  optimizer:
    type: adamw
    learning_rate: 0.0003
    weight_decay: 0.01
  batch_size: 12
  max_epochs: 50

constraints:
  allow_pretrained: false
  max_parameters: 30000000
  max_train_hours: 12
```

### PyTorch expansion

```python
# candidate/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LineAwareUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.e1 = Block(in_channels, c)
        self.e2 = Block(c, 2 * c)
        self.e3 = Block(2 * c, 4 * c)
        self.e4 = Block(4 * c, 8 * c)
        self.b = Block(8 * c, 16 * c)
        self.d4 = Block(16 * c + 8 * c, 8 * c)
        self.d3 = Block(8 * c + 4 * c, 4 * c)
        self.d2 = Block(4 * c + 2 * c, 2 * c)
        self.d1 = Block(2 * c + c, c)
        self.mask_head = nn.Conv2d(c, out_channels, kernel_size=1)
        self.line_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.d4(torch.cat([d4, e4], dim=1))
        d3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return {"mask_logits": self.mask_head(d1), "line_logits": self.line_head(d1)}


def build_model(input_spec, output_spec):
    return LineAwareUNet(input_spec["channels"], output_spec["classes"], base_channels=32)
```

---

## 11. Example architecture 4: small HRNet-like high-resolution segmentation model

For thin features, repeated downsampling can destroy signal. An HRNet-like model keeps a high-resolution branch alive while fusing lower-resolution contextual branches.

This is useful for contrails because it can preserve fine line geometry while still incorporating larger cloud/context cues.

### YAML representation

```yaml
name: tiny_hrnet_contrail_v1
task: binary_segmentation

input:
  modality: ground_camera_rgb
  tensor_shape: [batch, 3, 512, 512]
  sequence_length: 1

model:
  family: tiny_hrnet
  entrypoint: model:build_model
  params:
    in_channels: 3
    out_channels: 1
    widths: [32, 64, 128]
    stages: 3
    fusion: sum_after_projection
    norm: groupnorm

training:
  loss:
    type: focal_dice
    focal_gamma: 2.0
    dice_weight: 0.6
    focal_weight: 0.4
  optimizer:
    type: adamw
    learning_rate: 0.0002
    weight_decay: 0.02
  batch_size: 8
  max_epochs: 60

constraints:
  allow_pretrained: false
  max_parameters: 50000000
  max_train_hours: 16
```

### PyTorch expansion

```python
# candidate/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class TinyHRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, widths=(32, 64, 128)):
        super().__init__()
        w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, w1), w1),
            nn.ReLU(inplace=True),
            ResidualBlock(w1),
        )
        self.to_half = nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1)
        self.to_quarter = nn.Conv2d(w2, w3, kernel_size=3, stride=2, padding=1)
        self.high_blocks = nn.ModuleList([ResidualBlock(w1) for _ in range(3)])
        self.mid_blocks = nn.ModuleList([ResidualBlock(w2) for _ in range(3)])
        self.low_blocks = nn.ModuleList([ResidualBlock(w3) for _ in range(3)])
        self.mid_to_high = nn.Conv2d(w2, w1, kernel_size=1)
        self.low_to_high = nn.Conv2d(w3, w1, kernel_size=1)
        self.head = nn.Sequential(ResidualBlock(w1), nn.Conv2d(w1, out_channels, kernel_size=1))

    def forward(self, x):
        high = self.stem(x)
        mid = self.to_half(high)
        low = self.to_quarter(mid)
        for hb, mb, lb in zip(self.high_blocks, self.mid_blocks, self.low_blocks):
            high = hb(high)
            mid = mb(mid)
            low = lb(low)
            mid_up = F.interpolate(self.mid_to_high(mid), size=high.shape[-2:], mode="bilinear", align_corners=False)
            low_up = F.interpolate(self.low_to_high(low), size=high.shape[-2:], mode="bilinear", align_corners=False)
            high = high + mid_up + low_up
        return self.head(high)


def build_model(input_spec, output_spec):
    return TinyHRNet(input_spec["channels"], output_spec["classes"], widths=(32, 64, 128))
```

---

## 12. Example architecture 5: CNN encoder with attention bottleneck

Contrails may require long-range context: a line can extend across much of a frame, and local patches may be ambiguous against cloud. A CNN encoder with attention only at the bottleneck can be a useful compromise between UNet and a full transformer.

### YAML representation

```yaml
name: attention_bottleneck_unet_v1
task: binary_segmentation

input:
  modality: ground_camera_rgb
  tensor_shape: [batch, 3, 512, 512]
  sequence_length: 1

model:
  family: unet_with_attention_bottleneck
  entrypoint: model:build_model
  params:
    in_channels: 3
    out_channels: 1
    base_channels: 32
    bottleneck_channels: 256
    attention:
      type: multihead_self_attention_2d
      heads: 8
      apply_at_stride: 16

training:
  loss:
    type: bce_dice
    bce_weight: 0.4
    dice_weight: 0.6
  optimizer:
    type: adamw
    learning_rate: 0.0001
    weight_decay: 0.03
  batch_size: 6
  max_epochs: 60

constraints:
  allow_pretrained: false
  max_parameters: 60000000
  max_train_hours: 16
```

### PyTorch expansion

```python
# candidate/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBottleneck(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        y = self.norm(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + y
        tokens = tokens + self.ff(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.e1 = ConvBlock(in_channels, c)
        self.e2 = ConvBlock(c, 2 * c)
        self.e3 = ConvBlock(2 * c, 4 * c)
        self.e4 = ConvBlock(4 * c, 8 * c)
        self.attn = AttentionBottleneck(8 * c, heads=8)
        self.d3 = ConvBlock(8 * c + 4 * c, 4 * c)
        self.d2 = ConvBlock(4 * c + 2 * c, 2 * c)
        self.d1 = ConvBlock(2 * c + c, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.attn(e4)
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def build_model(input_spec, output_spec):
    return AttentionUNet(input_spec["channels"], output_spec["classes"], base_channels=32)
```

---

## 13. Candidate validation checklist

Before launching Docker, the trusted runner should validate:

### Manifest

- Required fields exist.
- Dataset name is allowed.
- Model entrypoint is allowed.
- Output contract is known.
- Training budget is within limits.
- No pretrained weights are requested.
- No arbitrary artifact paths are specified.

### Python source

Reject obvious hazards:

```text
import os
import sys
import subprocess
import socket
import requests
import urllib
import httpx
import pathlib
import shutil
import ctypes
import pickle
import dill
import cloudpickle
import importlib
```

Reject dangerous calls:

```text
open(...)
eval(...)
exec(...)
compile(...)
__import__(...)
getattr(__builtins__, ...)
torch.load(...)
```

Allowlist imports initially:

```text
torch
torch.nn
torch.nn.functional
math
typing
collections
dataclasses
```

Optionally allow:

```text
einops
timm
torchvision
```

but only if the trusted runner disables pretrained downloads and blocks network.

### Runtime contract

Inside the container, after importing the candidate:

1. Build the model with a controlled `input_spec` and `output_spec`.
2. Count parameters.
3. Run a CPU or GPU smoke test with synthetic data.
4. Check output shape.
5. Check output dtype.
6. Check that outputs are tensors or approved dictionaries of tensors.
7. Check one forward/backward pass.
8. Only then start the real training loop.

---

## 14. Worker entrypoint sketch

```python
# /runner/worker_entrypoint.py

from pathlib import Path
import argparse
import importlib.util
import json
import torch


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_test_model(model, input_spec, output_spec):
    channels = input_spec["channels"]
    height = input_spec["height"]
    width = input_spec["width"]
    sequence_length = input_spec.get("sequence_length", 1)

    if sequence_length > 1:
        x = torch.randn(2, sequence_length, channels, height, width)
    else:
        x = torch.randn(2, channels, height, width)

    y = model(x)

    if isinstance(y, dict):
        if "mask_logits" not in y:
            raise RuntimeError("Dictionary output must contain mask_logits")
        logits = y["mask_logits"]
    else:
        logits = y

    expected = (2, output_spec["classes"], height, width)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Bad output shape: got {tuple(logits.shape)}, expected {expected}")

    loss = logits.mean()
    loss.backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.manifest).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The real implementation should parse YAML and validate against schema.
    input_spec = {"channels": 3, "height": 512, "width": 512, "sequence_length": 1}
    output_spec = {"task": "binary_segmentation", "classes": 1}

    module = load_module_from_path(candidate_dir / "model.py")
    model = module.build_model(input_spec, output_spec)

    smoke_test_model(model, input_spec, output_spec)

    params = sum(p.numel() for p in model.parameters())
    (output_dir / "model_summary.json").write_text(json.dumps({"parameters": params}, indent=2))

    # Then call the trusted training loop:
    # train_model(model, input_spec, output_spec, output_dir)


if __name__ == "__main__":
    main()
```

The worker imports untrusted candidate code, but it does so inside the Docker container, not inside the trusted host runner process.

---

## 15. MLflow integration

Recommended pattern:

```text
Docker container:
  writes:
    /outputs/metrics.jsonl
    /outputs/final_metrics.json
    /outputs/model_summary.json
    /outputs/checkpoints/...
    /outputs/prediction_samples/...

Trusted runner:
  reads outputs
  logs metrics/artifacts to MLflow
```

Advantages:

- Candidate code never sees the MLflow write token.
- The container can run with `--network none`.
- Failed or rejected runs do not pollute MLflow unless the trusted runner chooses to log them.
- MLflow access can be made read-only for the agent.

Agent-facing MLflow tools should be narrow:

```bash
get_best_runs --experiment ground_camera_contrail --metric val/iou --limit 20
get_run_summary run-000042
get_run_artifacts run-000042 --kind prediction_samples
```

---

## 16. Secret handling

Secrets should live outside the agent workspace:

```text
/srv/contrail-agent/secrets/mlflow.env
```

Permissions:

```bash
chown runner-user:runner-group /srv/contrail-agent/secrets/mlflow.env
chmod 0600 /srv/contrail-agent/secrets/mlflow.env
```

Do not expose this file to:

- Pi.
- pi-enclave.
- candidate code.
- Docker containers, unless direct MLflow logging from the container is explicitly enabled.

If direct MLflow logging from the container is required, use a dedicated low-privilege MLflow token and inject only that token:

```bash
--env-file /srv/contrail-agent/secrets/mlflow-container-readwrite.env
```

---

## 17. Host wrapper commands

Expose a minimal command surface to the agent.

### submit_candidate

```bash
submit_candidate ./candidate
```

Responsibilities:

1. Resolve path inside workspace.
2. Copy candidate to a new run directory.
3. Validate manifest and source.
4. Launch Docker.
5. Capture logs.
6. Upload results to MLflow.
7. Print run ID.

### list_runs

```bash
list_runs --recent 20
```

### get_run_summary

```bash
get_run_summary run-000042
```

### get_best_runs

```bash
get_best_runs --metric val/iou --limit 20
```

### cancel_run

Optional, if training jobs are asynchronous.

```bash
cancel_run run-000042
```

If asynchronous execution is needed, the trusted runner can use:

- `systemd-run --user`
- a small local queue daemon
- a lockfile-based single-GPU queue
- `tmux`/`screen` only as a temporary manual bridge

Avoid letting the agent directly manage process lifecycles.

---

## 18. Docker image

A minimal Dockerfile sketch:

```Dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /runner

RUN pip install --no-cache-dir \
    numpy \
    pyyaml \
    scikit-image \
    opencv-python-headless \
    mlflow

COPY runner/ /runner/

ENV PYTHONUNBUFFERED=1
```

For old clusters, pin CUDA and PyTorch to versions compatible with the installed NVIDIA driver.

The trusted runner should record:

```text
docker image digest
NVIDIA driver version
CUDA runtime version
PyTorch version
git commit of runner
git commit/digest of candidate
```

---

## 19. Ground-camera contrail modeling notes

For the warm-up problem, the agent should explore architectures that address:

- thin, high-aspect-ratio features
- weak contrast against sky or cloud
- temporal persistence
- occlusion by cloud
- changing illumination
- compression artifacts in MP4 frames
- class imbalance
- uncertain labels

Useful families:

```text
baseline:
  - UNet
  - UNet++
  - FPN-style decoders

thin-feature preserving:
  - HRNet-like high-resolution networks
  - shallow high-resolution branches
  - auxiliary line/edge heads

temporal:
  - channel-stacked temporal UNet
  - 3D convolution stem followed by 2D decoder
  - ConvLSTM bottleneck
  - temporal attention over frame features

contextual:
  - attention bottleneck UNet
  - axial attention modules
  - lightweight transformer bottleneck

loss/training:
  - BCE + Dice
  - Focal + Dice
  - boundary/line auxiliary losses
  - hard-negative sampling
  - cloud-heavy negative mining
```

Initial policy should allow only `model.py`. Later, allow `transforms.py` and `loss.py` if the agent needs more research freedom.

---

## 20. Staged implementation plan

### Stage 1: Native runner skeleton

- Implement `submit_candidate`.
- Validate manifest.
- Copy candidate into run directory.
- Run smoke test.
- No Docker yet.
- No real training yet.

### Stage 2: Docker smoke test

- Build training image.
- Run candidate import and smoke test inside Docker.
- Mount candidate read-only.
- Mount output directory read/write.
- Use `--network none`.

### Stage 3: Real training

- Add fixed dataset loader.
- Add fixed training loop.
- Add metric output files.
- Add checkpoint writing.
- Add prediction sample artifacts.

### Stage 4: MLflow upload

- Trusted runner uploads outputs to MLflow.
- Agent receives run IDs and summaries.
- Agent can query best runs.

### Stage 5: Research expansion

- Allow temporal inputs.
- Allow auxiliary heads.
- Allow controlled custom losses.
- Add architecture-family-specific validation.
- Add timeouts and asynchronous queueing.

### Stage 6: Satellite/multispectral extension

- Replace RGB `input_spec` with multispectral band metadata.
- Allow non-RGB stems.
- Add band dropout and spectral normalization.
- Add temporal/geospatial context if available.
- Keep the same system boundary.

---

## 21. Bottom line

The recommended design is:

```text
pi-enclave/Gondolin protects the agent control plane.
Docker protects the GPU execution plane.
The trusted host runner connects them through a narrow, auditable interface.
MLflow is used as an observation layer, preferably written only by the trusted runner.
```

For the first ground-camera contrail experiments, keep candidate freedom mostly inside `model.py`. That is enough to explore useful segmentation architectures while keeping the trusted runner and Docker boundary simple.
