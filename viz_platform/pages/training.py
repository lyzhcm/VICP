""" Training Page - Launch & Monitor """
import streamlit as st
import subprocess
import sys
import os
import time
import json
import re
import threading
import queue
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "training_logs"
LOG_DIR.mkdir(exist_ok=True)


def render():
    st.title("Training Dashboard")

    if "training_status" not in st.session_state:
        st.session_state.training_status = "idle"
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    if "training_metrics" not in st.session_state:
        st.session_state.training_metrics = {"step": [], "loss": [], "id_loss": [], "icl_loss": [], "ot_loss": [], "std": []}
    if "training_process" not in st.session_state:
        st.session_state.training_process = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Panel")
        status = st.session_state.training_status

        status_colors = {"idle": "gray", "running": "green", "completed": "blue", "error": "red"}
        st.markdown(f"**Status:** :{status_colors.get(status, 'gray')}[{status.upper()}]")

        if status == "idle":
            if st.button("🚀 Launch Training", type="primary", use_container_width=True):
                launch_training()

        elif status == "running":
            if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                stop_training()

        elif status in ("completed", "error"):
            if st.button("🔄 Reset & Start New", use_container_width=True):
                st.session_state.training_status = "idle"
                st.session_state.training_logs = []
                st.session_state.training_metrics = {"step": [], "loss": [], "id_loss": [], "icl_loss": [], "ot_loss": [], "std": []}
                st.rerun()

        st.markdown("---")

        st.subheader("Metrics Summary")
        metrics = st.session_state.training_metrics
        if metrics["step"]:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Steps", len(metrics["step"]))
            if metrics["loss"]:
                m2.metric("Latest Loss", f"{metrics['loss'][-1]:.4f}")
            if metrics["std"]:
                m3.metric("Latest Std", f"{metrics['std'][-1]:.4f}")
        else:
            st.caption("No metrics recorded yet")

        st.markdown("---")
        st.subheader("Guide")
        st.markdown("""
        1. Set config in **⚙️ Configuration** tab
        2. Click **Launch Training** here
        3. Monitor losses in real-time
        4. See results in **📊 Results** tab
        """)

    with col2:
        tabs = st.tabs(["📈 Loss Curves", "📋 Training Log", "🔧 System Info"])

        with tabs[0]:
            render_loss_charts()

        with tabs[1]:
            st.subheader("Live Logs")
            log_container = st.container()
            with log_container:
                for line in st.session_state.training_logs[-100:]:
                    st.text(line)

        with tabs[2]:
            render_system_info()

    if status == "running":
        poll_training_progress()
        time.sleep(0.5)
        st.rerun()


def launch_training():
    from pages.config import get_current_config, build_command

    cfg = get_current_config()
    log_file = LOG_DIR / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"

    os.environ["WANDB_MODE"] = "disabled"

    project_root = Path(__file__).parent.parent.parent
    cmd = [
        sys.executable, "-u",
        str(project_root / "train_vpt_lora.py"),
        f"--dataset_name={cfg['dataset_name']}",
        f"--cluster_index={cfg['cluster_index']}",
        f"--vision_model={cfg['vision_model']}",
        f"--llm_model={cfg['llm_model']}",
        f"--num_id_tokens={cfg['num_id_tokens']}",
        f"--num_vpt_tokens={cfg['num_vpt_tokens']}",
        f"--num_icl_samples={cfg['num_icl_samples']}",
        f"--num_icl_bs={cfg['num_icl_bs']}",
        f"--icl_loss_weight={cfg['icl_loss_weight']}",
        f"--ot_loss_weight={cfg['ot_loss_weight']}",
        f"--learning_rate={cfg['learning_rate']}",
        f"--weight_decay={cfg['weight_decay']}",
        f"--per_device_train_batch_size={cfg['per_device_train_batch_size']}",
        f"--max_steps={cfg['max_steps']}",
        f"--lr_scheduler_type={cfg['lr_scheduler_type']}",
        f"--gradient_accumulation_steps={cfg['gradient_accumulation_steps']}",
        f"--max_grad_norm={cfg['max_grad_norm']}",
        f"--save_steps={cfg['save_steps']}",
        f"--eval_steps={cfg['eval_steps']}",
        f"--logging_steps={cfg['logging_steps']}",
        f"--dataloader_num_workers={cfg['dataloader_num_workers']}",
        f"--save_total_limit={cfg['save_total_limit']}",
        f"--output_dir={cfg['output_dir']}",
        "--save_safetensors=False",
        "--ddp_find_unused_parameters=False",
        "--report_to=none",
        "--dataloader_drop_last=True",
        "--dataloader_pin_memory=False",
        "--dataloader_persistent_workers=True",
    ]
    if cfg.get("fp16", True):
        cmd.append("--fp16=True")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root),
        )
        st.session_state.training_process = process
        st.session_state.training_status = "running"
        st.session_state.training_logs = [f"[System] Training launched at {time.strftime('%H:%M:%S')}"]
        st.session_state.current_log_file = str(log_file)
        st.session_state.training_metrics = {"step": [], "loss": [], "id_loss": [], "icl_loss": [], "ot_loss": [], "std": []}

        with open(log_file, "w") as f:
            f.write(f"Training log - {time.ctime()}\n{'='*60}\n")

    except Exception as e:
        st.session_state.training_status = "error"
        st.session_state.training_logs.append(f"[ERROR] Failed to launch: {str(e)}")


def stop_training():
    if st.session_state.training_process:
        st.session_state.training_process.terminate()
        st.session_state.training_status = "idle"
        st.session_state.training_logs.append("[System] Training stopped by user")


def poll_training_progress():
    process = st.session_state.training_process
    if process is None:
        return

    exit_code = process.poll()
    if exit_code is not None:
        remaining = process.stdout.read() if process.stdout else ""
        if remaining:
            for line in remaining.strip().split("\n"):
                if line.strip():
                    st.session_state.training_logs.append(line)
        if exit_code == 0:
            st.session_state.training_status = "completed"
            st.session_state.training_logs.append("[System] Training completed successfully!")
        else:
            st.session_state.training_status = "error"
            st.session_state.training_logs.append(f"[System] Training failed with exit code {exit_code}")
        return

    line_buffer = []
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line:
                st.session_state.training_logs.append(line)
                parse_metric_line(line)
                if hasattr(st.session_state, "current_log_file"):
                    with open(st.session_state.current_log_file, "a") as f:
                        f.write(line + "\n")
    except Exception:
        pass


def parse_metric_line(line):
    pattern = r"\{.*'loss':\s*([\d.]+).*'id_loss':\s*([\d.]+).*'icl_loss':\s*([\d.]+).*'ot_loss':\s*([\d.e\-+]+).*'std':\s*([\d.e\-+]+)"
    match = re.search(pattern, line)
    if match:
        step = len(st.session_state.training_metrics["step"])
        st.session_state.training_metrics["step"].append(step)
        st.session_state.training_metrics["loss"].append(float(match.group(1)))
        st.session_state.training_metrics["id_loss"].append(float(match.group(2)))
        st.session_state.training_metrics["icl_loss"].append(float(match.group(3)))
        st.session_state.training_metrics["ot_loss"].append(float(match.group(4)))
        st.session_state.training_metrics["std"].append(float(match.group(5)))
        return

    pattern2 = r"'loss':\s*([\d.]+)"
    match2 = re.search(pattern2, line)
    if match2 and re.search(r"'grad_norm'|'learning_rate'", line):
        return

    step_match = re.search(r"Step\s*(\d+).*(loss|Loss).*?([\d.]+)", line, re.IGNORECASE)
    if step_match:
        step = len(st.session_state.training_metrics["step"])
        st.session_state.training_metrics["step"].append(step)
        st.session_state.training_metrics["loss"].append(float(step_match.group(3)))


def render_loss_charts():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = st.session_state.training_metrics

    if not metrics["step"]:
        st.info("No training data yet. Launch training to see loss curves.")
        return

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Total Loss", "Component Losses", "Feature Std (Collapse Monitor)"),
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(x=metrics["step"], y=metrics["loss"], mode="lines", name="Total Loss",
                   line=dict(color="#636EFA", width=2)), row=1, col=1
    )

    if any(metrics["id_loss"]):
        fig.add_trace(
            go.Scatter(x=metrics["step"], y=metrics["id_loss"], mode="lines", name="ID Loss (Triplet)",
                       line=dict(color="#EF553B")), row=2, col=1
        )
    if any(metrics["icl_loss"]):
        fig.add_trace(
            go.Scatter(x=metrics["step"], y=metrics["icl_loss"], mode="lines", name="ICL Loss (LLM)",
                       line=dict(color="#00CC96")), row=2, col=1
        )
    if any(metrics["ot_loss"]):
        fig.add_trace(
            go.Scatter(x=metrics["step"], y=metrics["ot_loss"], mode="lines", name="OT Loss (WPA)",
                       line=dict(color="#AB63FA")), row=2, col=1
        )

    if any(metrics["std"]):
        fig.add_trace(
            go.Scatter(x=metrics["step"], y=metrics["std"], mode="lines", name="Feature Std",
                       fill="tozeroy", line=dict(color="#FFA15A", width=2)), row=3, col=1
        )

    fig.update_layout(
        height=600, showlegend=True, template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Std", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_system_info():
    st.subheader("Hardware")
    c1, c2 = st.columns(2)
    with c1:
        try:
            import torch
            st.info(f"**PyTorch:** {torch.__version__}")
            st.info(f"**CUDA Available:** {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    st.info(f"**GPU {i}:** {torch.cuda.get_device_name(i)}")
        except ImportError:
            st.warning("PyTorch not found")
    with c2:
        import platform
        st.info(f"**Python:** {platform.python_version()}")
        st.info(f"**OS:** {platform.system()} {platform.release()}")

        try:
            import transformers
            st.info(f"**Transformers:** {transformers.__version__}")
        except ImportError:
            st.warning("Transformers not found")

    st.subheader("Estimated VRAM Usage")
    st.markdown("""
    | Component | VRAM |
    |-----------|------|
    | DINOv2 ViT-B/14 | ~350 MB |
    | Qwen3-0.6B | ~1.2 GB |
    | LoRA + Connectors | ~100 MB |
    | **Total (FP16)** | **~2 GB** |
    | **Training (batch 256)** | **~8-12 GB** |
    """)
