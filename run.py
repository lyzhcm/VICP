#!/usr/bin/env python
"""VICP Training Launcher - Terminal-based experiment runner."""
import os
import sys
import subprocess
import argparse
import time
import re
from pathlib import Path


BANNER = r"""
╔══════════════════════════════════════════════╗
║  VICP Training Platform v1.0                 ║
║  Visual In-Context Prompting                 ║
║  ICCV 2025 · Generalizable Object ReID       ║
╚══════════════════════════════════════════════╝
"""

PRESETS = {
    "quick": {
        "desc": "Quick test (100 steps, bs=64) - for verifying setup",
        "per_device_train_batch_size": 64,
        "max_steps": 100,
        "eval_steps": 50,
        "fp16": True,
        "dataloader_num_workers": 0,
        "output_dir": "./checkpoints/vicp_quick",
    },
    "standard": {
        "desc": "Standard training (5000 steps, bs=256) - paper default",
        "per_device_train_batch_size": 256,
        "max_steps": 5000,
        "eval_steps": 20,
        "fp16": True,
        "dataloader_num_workers": 4,
        "output_dir": "./checkpoints/vicp_standard",
    },
    "full": {
        "desc": "Full training (20000 steps, bs=256) - best results",
        "per_device_train_batch_size": 256,
        "max_steps": 20000,
        "eval_steps": 20,
        "fp16": True,
        "dataloader_num_workers": 8,
        "output_dir": "./checkpoints/vicp_full",
    },
}

DEFAULT_ARGS = {
    "dataset_name": "amazon",
    "cluster_index": 0,
    "vision_model": "dinov2_vitb14",
    "llm_model": "Qwen/Qwen3-0.6B",
    "num_id_tokens": 32,
    "num_vpt_tokens": 32,
    "num_icl_samples": 64,
    "num_icl_bs": 1,
    "icl_loss_weight": 1.0,
    "ot_loss_weight": 0.01,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "lr_scheduler_type": "constant",
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 0,
    "save_steps": 1000,
    "logging_steps": 1,
    "save_total_limit": 10,
    "save_safetensors": "False",
    "ddp_find_unused_parameters": "False",
    "report_to": "none",
    "dataloader_drop_last": "True",
    "dataloader_pin_memory": "False",
}


def check_environment():
    print("[*] Checking environment...")
    issues = []

    try:
        import torch
        print(f"    PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            issues.append("CUDA not available - training will be very slow on CPU")
    except ImportError:
        issues.append("PyTorch not found - install: pip install torch")

    try:
        import transformers
        print(f"    Transformers {transformers.__version__}")
    except ImportError:
        issues.append("Transformers not found - install: pip install transformers")

    data_root = "../groundingdino_cropped"
    if Path(data_root).exists():
        split_dir = Path(data_root) / "split"
        if split_dir.exists():
            cats = [d.name for d in split_dir.iterdir() if d.is_dir()]
            print(f"    Dataset found: {len(cats)} categories in {data_root}")
        else:
            issues.append(f"Dataset split dir not found at {data_root}/split/")
    else:
        issues.append(f"Dataset not found at {data_root}")
        issues.append("  Download: https://drive.google.com/drive/folders/1ubm0oo8-5wXLocoHIk5yt1CtzgXTg_1h")

    if issues:
        print("\n[!] Issues found:")
        for i in issues:
            print(f"    {i}")
        return False
    print("[+] Environment OK\n")
    return True


def build_command(args_dict):
    parts = ["python", "-u", "train_vpt_lora.py"]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k}={v}")
    return parts


def run_training(cmd_parts):
    print("[*] Launching training...")
    print("    " + " ".join(cmd_parts[:3]) + " ... (see below for full args)")
    print("-" * 60)

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    metrics = {"step": [], "loss": [], "id_loss": [], "icl_loss": [], "ot_loss": [], "std": []}
    eval_results = []

    try:
        proc = subprocess.Popen(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            print(line)

            m = re.search(r"'loss':\s*([\d.]+).*'id_loss':\s*([\d.]+).*'icl_loss':\s*([\d.]+).*'ot_loss':\s*([\de\-\+.]+).*'std':\s*([\de\-\+.]+)", line)
            if m:
                metrics["step"].append(len(metrics["step"]))
                metrics["loss"].append(float(m.group(1)))
                metrics["id_loss"].append(float(m.group(2)))
                metrics["icl_loss"].append(float(m.group(3)))
                metrics["ot_loss"].append(float(m.group(4)))
                metrics["std"].append(float(m.group(5)))

        rc = proc.poll()
        print("-" * 60)
        if rc == 0:
            print("[+] Training completed successfully!")
        else:
            print(f"[!] Training exited with code {rc}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    return metrics


def print_summary(metrics):
    if not metrics["step"]:
        return
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    n = len(metrics["step"])
    print(f"  Total steps: {n}")
    if metrics["loss"]:
        print(f"  Final loss:  {metrics['loss'][-1]:.4f}")
    if metrics["id_loss"]:
        print(f"  Final ID loss:  {metrics['id_loss'][-1]:.4f}")
    if metrics["icl_loss"]:
        print(f"  Final ICL loss: {metrics['icl_loss'][-1]:.4f}")
    if metrics["ot_loss"]:
        print(f"  Final OT loss:  {metrics['ot_loss'][-1]:.4f}")
    if metrics["std"]:
        print(f"  Final std:      {metrics['std'][-1]:.4f}")
    print("=" * 60)


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="VICP Training Launcher")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="Use a preset configuration")
    parser.add_argument("--split", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Cross-validation split (0-4)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable FP16")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--check-only", action="store_true", help="Only check environment, don't train")
    args = parser.parse_args()

    if args.check_only:
        check_environment()
        return

    if not check_environment():
        print("\n[!] Please fix the issues above before training.")
        return

    cfg = DEFAULT_ARGS.copy()

    if args.preset:
        preset = PRESETS[args.preset]
        print(f"[*] Using preset: {args.preset} - {preset['desc']}")
        for k, v in preset.items():
            if k != "desc":
                cfg[k] = v

    if args.split is not None:
        cfg["cluster_index"] = args.split
    if args.batch_size is not None:
        cfg["per_device_train_batch_size"] = args.batch_size
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.no_fp16:
        cfg["fp16"] = False
    elif args.fp16:
        cfg["fp16"] = True

    if int(cfg.get("dataloader_num_workers", 0)) == 0:
        cfg["dataloader_persistent_workers"] = "False"

    print("[*] Configuration:")
    for k, v in cfg.items():
        print(f"    {k}: {v}")

    cmd = build_command(cfg)

    print("\n[*] Full command:")
    print("    " + " \\\n      ".join(cmd))
    print()

    try:
        confirm = input("[?] Launch training? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[!] Cancelled.")
        return

    if confirm and confirm not in ("y", "yes", ""):
        print("[!] Cancelled.")
        return

    print()
    metrics = run_training(cmd)
    print_summary(metrics)


if __name__ == "__main__":
    main()
