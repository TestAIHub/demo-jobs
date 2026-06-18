"""
mock_training_job.py
--------------------
Simulates the console output of a real ML model training job.
Designed for Renku job feature demos — prints realistic logs with
timed delays so it looks and feels like a live training run.

Usage examples
--------------
# Defaults
python mock_training_job.py

# Custom run
python mock_training_job.py \\
    --model resnet101 \\
    --dataset cifar100 \\
    --epochs 10 \\
    --batch-size 64 \\
    --lr 5e-4 \\
    --optimizer sgd \\
    --device cpu \\
    --seed 7
"""

import argparse
import math
import random
import time

# ── CLI ───────────────────────────────────────────────────────────────────────

MODELS = {
    "resnet50":   {"total_params": 25_557_034, "trainable_params":  2_621_962, "size_mb":  97.4},
    "resnet101":  {"total_params": 44_549_160, "trainable_params":  4_718_592, "size_mb": 170.5},
    "efficientnet": {"total_params": 6_520_040, "trainable_params": 1_280_000, "size_mb":  29.6},
    "vit":        {"total_params": 86_567_656, "trainable_params": 12_582_912, "size_mb": 330.2},
}

DATASETS = {
    "imagenet-mini": {"classes": 10,  "train": 10_400, "val": 1_300, "test": 1_300, "steps": 325},
    "cifar10":       {"classes": 10,  "train": 50_000, "val": 5_000, "test": 5_000, "steps": 782},
    "cifar100":      {"classes": 100, "train": 50_000, "val": 5_000, "test": 5_000, "steps": 782},
    "flowers102":    {"classes": 102, "train":  6_149, "val":  1_020, "test": 1_020, "steps":  96},
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mock ML training job for Renku demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      choices=list(MODELS.keys()),   default="resnet50",
                   help="Model architecture to train")
    p.add_argument("--dataset",    choices=list(DATASETS.keys()), default="imagenet-mini",
                   help="Dataset to train on")
    p.add_argument("--epochs",     type=int,   default=5,         help="Number of training epochs")
    p.add_argument("--batch-size", type=int,   default=32,        help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=1e-3,      help="Initial learning rate")
    p.add_argument("--optimizer",  choices=["adamw", "sgd", "adam"], default="adamw",
                   help="Optimiser")
    p.add_argument("--device",     choices=["cuda", "cpu"],       default="cuda",
                   help="Compute device")
    p.add_argument("--seed",       type=int,   default=42,        help="Random seed")
    p.add_argument("--no-pretrain", action="store_true",
                   help="Train from scratch instead of using ImageNet weights")
    return p.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str, delay: float = 0.0):
    print(msg, flush=True)
    if delay:
        time.sleep(delay)

def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {100 * current / total:5.1f}%"

# ── Phase 1 — Environment ─────────────────────────────────────────────────────

def phase_environment(cfg: argparse.Namespace):
    import datetime
    run_id = datetime.datetime.now().strftime("run-%Y%m%d-%H%M")
    log("\n" + "=" * 65)
    log(f"  🚀  RENKU JOB  |  {cfg.model}-{cfg.dataset}  |  {run_id}")
    log("=" * 65)
    log("")
    log("[INFO] Initialising environment ...", delay=0.6)
    log("[INFO] Python        3.11.9")
    log("[INFO] PyTorch       2.3.1+cu121")
    if cfg.device == "cuda":
        log("[INFO] CUDA          available  →  NVIDIA A100 40 GB")
    else:
        log("[INFO] CUDA          not requested  →  running on CPU")
    log(f"[INFO] Seed          set to {cfg.seed} (torch + numpy + random)", delay=0.4)
    log(f"[INFO] Output dir    /outputs/{run_id}/")
    log("")

# ── Phase 2 — Data ────────────────────────────────────────────────────────────

def phase_data(cfg: argparse.Namespace) -> dict:
    ds = DATASETS[cfg.dataset]
    total = ds["train"] + ds["val"] + ds["test"]
    log(f"[DATA] Loading dataset: {cfg.dataset}  ({ds['classes']} classes, {total:,} samples)")
    log("[DATA] Scanning data directory ...", delay=1.2)
    log(f"[DATA] ✔  Train      {ds['train']:>6,} images  (80 %)")
    log(f"[DATA] ✔  Validation {ds['val']:>6,} images  (10 %)")
    log(f"[DATA] ✔  Test       {ds['test']:>6,} images  (10 %)")
    log("[DATA] Applying augmentations: RandomCrop(224), HorizontalFlip, ColorJitter")
    log(f"[DATA] DataLoaders ready  |  batch_size={cfg.batch_size}  |  num_workers=4", delay=0.5)
    log("")
    return ds

# ── Phase 3 — Model ───────────────────────────────────────────────────────────

def phase_model(cfg: argparse.Namespace, ds: dict):
    m = MODELS[cfg.model]
    log(f"[MODEL] Building {cfg.model.upper()} architecture ...")
    time.sleep(0.8)
    if cfg.no_pretrain:
        log("[MODEL] Training from scratch (--no-pretrain)", delay=0.4)
    else:
        log("[MODEL] Loading ImageNet pre-trained weights from torchvision hub ...", delay=1.0)
    log(f"[MODEL] Replacing classifier head → {ds['classes']} output classes")
    if not cfg.no_pretrain:
        log("[MODEL] Freezing backbone layers (early conv blocks)")
    log(f"[MODEL] Trainable parameters  : {m['trainable_params']:>12,}")
    log(f"[MODEL] Total parameters      : {m['total_params']:>12,}")
    log(f"[MODEL] Model moved to {cfg.device.upper()}", delay=0.3)

    # Optimizer description
    opt_details = {
        "adamw": f"AdamW  (lr={cfg.lr:.2e}, weight_decay=1e-4)",
        "adam":  f"Adam   (lr={cfg.lr:.2e})",
        "sgd":   f"SGD    (lr={cfg.lr:.2e}, momentum=0.9, weight_decay=5e-4)",
    }
    log(f"[MODEL] Optimizer  : {opt_details[cfg.optimizer]}")
    log(f"[MODEL] Scheduler  : CosineAnnealingLR  (T_max={cfg.epochs})")
    log("[MODEL] Loss       : CrossEntropyLoss")
    log("")

# ── Phase 4 — Training loop ───────────────────────────────────────────────────

def phase_training(cfg: argparse.Namespace, ds: dict) -> float:
    epochs = cfg.epochs
    steps  = ds["steps"]

    log("[TRAIN] Starting training loop ...")
    log(f"[TRAIN] Epochs: {epochs}  |  Steps/epoch: {steps}")
    log("")

    # Generate plausible converging curves for any epoch count
    def curve(start, end, n):
        return [end + (start - end) * math.exp(-3.5 * i / max(n - 1, 1)) for i in range(n)]

    # Slightly harder to converge when training from scratch
    acc_ceiling = 0.76 if cfg.no_pretrain else 0.83
    base_loss  = curve(1.95, 0.55, epochs)
    base_acc   = curve(0.38, acc_ceiling, epochs)
    base_vloss = curve(1.60, 0.62, epochs)
    base_vacc  = curve(0.46, acc_ceiling + 0.02, epochs)
    lr_values  = [cfg.lr * math.cos(i / max(epochs - 1, 1) * math.pi / 2) for i in range(epochs)]

    best_val_acc = 0.0
    # Epoch timing: spread total ~30 s across epochs; more epochs → less per epoch
    epoch_duration = max(3.0, min(6.0, 30.0 / epochs))
    step_delay = epoch_duration / steps

    for epoch in range(1, epochs + 1):
        log(f"── Epoch {epoch}/{epochs} " + "─" * 46)
        loss = base_loss[epoch - 1]
        acc  = base_acc[epoch - 1]

        checkpoints = {int(steps * p) for p in (0.25, 0.5, 0.75, 1.0)}
        for step in range(1, steps + 1):
            noise_l = random.uniform(-0.012, 0.012)
            noise_a = random.uniform(-0.008, 0.008)
            step_loss = max(0.1, loss + noise_l + 0.15 * math.cos(step / steps * math.pi))
            step_acc  = min(1.0, acc + noise_a)
            time.sleep(step_delay)
            if step in checkpoints:
                bar = progress_bar(step, steps)
                log(f"  Step {step:>4}/{steps}  {bar}  loss={step_loss:.4f}  acc={step_acc:.4f}")

        log("  → Validation ...", delay=0.8)
        val_loss = base_vloss[epoch - 1] + random.uniform(-0.01, 0.01)
        val_acc  = base_vacc[epoch - 1]  + random.uniform(-0.005, 0.005)
        log(f"  → val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_values[epoch-1]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            log(f"  ✔  New best model saved  →  checkpoint_epoch{epoch:02d}.pt")
        log("")

    return best_val_acc

# ── Phase 5 — Evaluation ─────────────────────────────────────────────────────

def phase_evaluation(cfg: argparse.Namespace, ds: dict):
    log("[EVAL] Loading best checkpoint for final test evaluation ...", delay=0.8)
    log(f"[EVAL] Running inference on {ds['test']:,} test samples ...", delay=1.5)

    # Generate per-class accuracies (100 classes → show top/bottom 5 summary)
    n_classes = ds["classes"]
    random.seed(cfg.seed + 1)
    class_accs = [round(random.uniform(0.62, 0.95), 2) for _ in range(n_classes)]

    log("")
    if n_classes <= 12:
        # Show all classes
        class_names = {
            10:  ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
            102: [f"class_{i:03d}" for i in range(102)],
        }.get(n_classes, [f"class_{i:03d}" for i in range(n_classes)])
        log("  Class-level accuracy:")
        for name, acc in zip(class_names, class_accs):
            bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
            log(f"    {name:<14}  {bar}  {acc:.2f}")
    else:
        # Summarise for large class counts
        sorted_accs = sorted(enumerate(class_accs), key=lambda x: x[1])
        log(f"  {n_classes} classes — showing 5 best / 5 worst:")
        log("  Best:")
        for idx, acc in sorted_accs[-5:][::-1]:
            log(f"    class_{idx:03d}      {'▓' * int(acc * 20)}  {acc:.2f}")
        log("  Worst:")
        for idx, acc in sorted_accs[:5]:
            log(f"    class_{idx:03d}      {'▓' * int(acc * 20)}  {acc:.2f}")

    overall = sum(class_accs) / len(class_accs)
    log("")
    log(f"  Overall test accuracy : {overall:.4f}")
    log(f"  Top-5 test accuracy   : {min(1.0, overall + 0.19):.4f}")
    log(f"  Test loss             : {0.38 + random.uniform(-0.03, 0.03):.4f}")
    log("")

# ── Phase 6 — Wrap-up ─────────────────────────────────────────────────────────

def phase_wrapup(cfg: argparse.Namespace, best_val_acc: float, start_time: float):
    elapsed = time.time() - start_time
    size_mb = MODELS[cfg.model]["size_mb"]
    log("[DONE] Saving final model artefacts ...")
    time.sleep(0.5)
    log(f"[DONE] ✔  model_final.pt              ({size_mb} MB)")
    log("[DONE] ✔  training_curves.png")
    log("[DONE] ✔  confusion_matrix.png")
    log("[DONE] ✔  metrics.json")
    log("")
    log("=" * 65)
    log("  ✅  Training complete")
    log(f"  Model             : {cfg.model}")
    log(f"  Dataset           : {cfg.dataset}")
    log(f"  Epochs            : {cfg.epochs}")
    log(f"  Batch size        : {cfg.batch_size}")
    log(f"  Learning rate     : {cfg.lr:.2e}")
    log(f"  Optimizer         : {cfg.optimizer}")
    log(f"  Best val accuracy : {best_val_acc:.4f}")
    log(f"  Total runtime     : {elapsed:.1f} s")
    log("=" * 65)
    log("")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg   = parse_args()
    start = time.time()
    random.seed(cfg.seed)

    phase_environment(cfg)
    ds = phase_data(cfg)
    phase_model(cfg, ds)
    best_val_acc = phase_training(cfg, ds)
    phase_evaluation(cfg, ds)
    phase_wrapup(cfg, best_val_acc, start)

if __name__ == "__main__":
    main()
