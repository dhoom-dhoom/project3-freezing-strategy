"""
Project 3 training script for IndiWASTE.

Karpathy-style autoresearch adaptation:
- fixed 300-second budget
- fixed dataset/eval harness in prepare.py
- single intended search surface: freeze strategy
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from prepare import (
    DATASET_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    TIME_BUDGET,
    evaluate_classifier,
    make_dataloaders,
    save_eval_artifacts,
)

# ---------------------------------------------------------------------------
# Mutable experiment surface
# ---------------------------------------------------------------------------

FREEZE_STRATEGY = "all_but_head"

# Allowed values:
# - all_but_head
# - freeze_early
# - freeze_late
# - freeze_none

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

MODEL_NAME = "resnet18_imagenet"
LOCAL_RESNET18_WEIGHTS = "resnet18-f37072fd.pth"
HEAD_LR = 1e-3
BACKBONE_LR = 2e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = DEFAULT_BATCH_SIZE
EVAL_BATCH_SIZE = DEFAULT_EVAL_BATCH_SIZE
GRAD_CLIP_NORM = 1.0
SMOKE_TEST_TIME_BUDGET = 15
SMOKE_TEST_MAX_STEPS = 3


@dataclass
class RuntimeConfig:
    device: torch.device
    device_type: str
    amp_enabled: bool


def detect_runtime():
    cuda_available = torch.cuda.is_available()
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_version: {getattr(torch.version, 'cuda', None)}")
    print(f"cuda_available: {cuda_available}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")

    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        probe_text = (probe.stdout or probe.stderr or "").strip()
        if probe_text:
            print(f"nvidia_smi: {probe_text}")
    except Exception as exc:
        print(f"nvidia_smi: unavailable ({type(exc).__name__})")

    require_cuda = os.environ.get("REQUIRE_CUDA", "0") == "1"
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required for this run, but no GPU is available.")

    if cuda_available:
        device = torch.device("cuda")
        amp_enabled = True
    else:
        device = torch.device("cpu")
        amp_enabled = False
    return RuntimeConfig(device=device, device_type=device.type, amp_enabled=amp_enabled)


def build_model():
    local_override = os.environ.get("RESNET18_WEIGHTS_PATH")
    candidate_paths = []
    if local_override:
        candidate_paths.append(Path(local_override).expanduser())
    candidate_paths.append(Path(__file__).resolve().parent / LOCAL_RESNET18_WEIGHTS)

    for weights_path in candidate_paths:
        if weights_path.exists():
            model = resnet18(weights=None, progress=False)
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 10)
            print(f"Loaded local pretrained weights from {weights_path}")
            return model

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights, progress=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    return model


def apply_freeze_strategy(model, strategy: str):
    if strategy not in {"all_but_head", "freeze_early", "freeze_late", "freeze_none"}:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    for param in model.parameters():
        param.requires_grad = False

    if strategy == "all_but_head":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif strategy == "freeze_early":
        for module in (model.layer3, model.layer4, model.fc):
            for param in module.parameters():
                param.requires_grad = True
    elif strategy == "freeze_late":
        for module in (model.conv1, model.bn1, model.layer1, model.layer2, model.fc):
            for param in module.parameters():
                param.requires_grad = True
    elif strategy == "freeze_none":
        for param in model.parameters():
            param.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer(model):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY}
        )
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY}
        )
    if not param_groups:
        raise RuntimeError("Freeze strategy left no trainable parameters.")
    return torch.optim.AdamW(param_groups)


def _autocast_context(runtime: RuntimeConfig):
    if runtime.amp_enabled:
        return torch.amp.autocast(device_type=runtime.device_type, dtype=torch.float16)
    return contextlib.nullcontext()


def _save_pre_eval_checkpoint(model):
    try:
        torch.save(model.state_dict(), "checkpoint_pre_eval.pt")
        print("Saved checkpoint_pre_eval.pt")
    except Exception as exc:
        print(f"Warning: could not save checkpoint_pre_eval.pt: {exc}")


def _run_training_once(runtime: RuntimeConfig, smoke_test: bool):
    t_start = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model().to(runtime.device)
    apply_freeze_strategy(model, FREEZE_STRATEGY)
    total_params, trainable_params = count_parameters(model)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, _ = make_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        smoke_test=smoke_test,
    )

    time_budget = SMOKE_TEST_TIME_BUDGET if smoke_test else TIME_BUDGET
    max_steps = SMOKE_TEST_MAX_STEPS if smoke_test else None
    model.train()
    step = 0
    running_loss = 0.0
    last_log_time = 0.0
    t_train_start = time.time()
    train_iter = iter(train_loader)

    print(f"Model: {MODEL_NAME}")
    print(f"Freeze strategy: {FREEZE_STRATEGY}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Time budget: {time_budget}s")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    while True:
        now = time.time()
        training_seconds = now - t_train_start
        if training_seconds >= time_budget:
            break
        if max_steps is not None and step >= max_steps:
            break

        try:
            images, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, targets = next(train_iter)

        images = images.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(runtime):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            GRAD_CLIP_NORM,
        )
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if time.time() - last_log_time >= 5 or step == 1:
            avg_loss = running_loss / max(step, 1)
            elapsed = time.time() - t_train_start
            remaining = max(0.0, time_budget - elapsed)
            print(
                f"step {step:04d} | train_loss: {avg_loss:.4f} | "
                f"elapsed: {elapsed:.1f}s | remaining: {remaining:.1f}s"
            )
            last_log_time = time.time()

    total_training_time = time.time() - t_train_start
    _save_pre_eval_checkpoint(model)
    val_result = evaluate_classifier(model, val_loader, runtime.device, criterion)
    save_eval_artifacts(val_result)

    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = float("nan")

    total_seconds = time.time() - t_start
    return {
        "model": model,
        "step": step,
        "training_seconds": total_training_time,
        "total_seconds": total_seconds,
        "peak_memory_mb": peak_memory_mb,
        "val_result": val_result,
        "num_params": total_params,
        "trainable_params": trainable_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Project 3 autoresearch training script")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short validation pass.")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"ignored_cli_args: {unknown_args}")

    runtime = detect_runtime()
    print(f"Device: {runtime.device}")
    print(f"AMP enabled: {runtime.amp_enabled}")

    result = _run_training_once(runtime=runtime, smoke_test=args.smoke_test)
    val_result = result["val_result"]

    print("---")
    print(f"val_error:         {val_result.val_error:.6f}")
    print(f"val_macro_f1:      {val_result.val_macro_f1:.6f}")
    print(f"val_accuracy:      {val_result.val_accuracy:.6f}")
    print(f"val_loss:          {val_result.val_loss:.6f}")
    print(f"training_seconds:  {result['training_seconds']:.1f}")
    print(f"total_seconds:     {result['total_seconds']:.1f}")
    if result["peak_memory_mb"] == result["peak_memory_mb"]:
        print(f"peak_memory_mb:    {result['peak_memory_mb']:.1f}")
    else:
        print("peak_memory_mb:    n/a")
    print(f"num_steps:         {result['step']}")
    print(f"num_params_M:      {result['num_params'] / 1e6:.2f}")
    print(f"trainable_params_M: {result['trainable_params'] / 1e6:.2f}")
    print(f"freeze_strategy:   {FREEZE_STRATEGY}")
    print(f"model_name:        {MODEL_NAME}")
    if args.smoke_test:
        print("smoke_test:        true")

    del result["model"]
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
