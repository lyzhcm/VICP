""" Configuration Page - Experiment Settings """
import streamlit as st
import os
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).parent.parent / "experiments"
CONFIG_DIR.mkdir(exist_ok=True)


def render():
    st.title("Experiment Configuration")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Presets")
        if st.button("📥 Load Default Config", use_container_width=True):
            load_default_config()
        if st.button("💾 Save Current Config", use_container_width=True):
            save_config()
        st.markdown("---")
        saved_configs = list(CONFIG_DIR.glob("*.json"))
        if saved_configs:
            st.subheader("Saved Configs")
            for cfg in saved_configs:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.text(cfg.stem)
                with c2:
                    if st.button("📂", key=f"load_{cfg.stem}", help=f"Load {cfg.stem}"):
                        load_saved_config(cfg)
                    st.markdown("")

    with col2:
        default_cfg = {
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
            "per_device_train_batch_size": 256,
            "max_steps": 5000,
            "fp16": True,
            "lr_scheduler_type": "constant",
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 0,
            "save_steps": 1000,
            "eval_steps": 20,
            "logging_steps": 1,
            "dataloader_num_workers": 16,
            "dataloader_persistent_workers": True,
            "dataloader_pin_memory": False,
            "dataloader_drop_last": True,
            "save_total_limit": 10,
            "save_safetensors": False,
            "ddp_find_unused_parameters": False,
            "report_to": "none",
            "output_dir": "./checkpoints/vicp_exp",
        }

        tabs = st.tabs(["📂 Data", "🏗️ Model", "🎯 Training", "🚀 Advanced"])

        with tabs[0]:
            st.subheader("Data Settings")
            st.session_state.cfg_dataset_name = st.selectbox(
                "Dataset", ["amazon"],
                index=0,
                key="cfg_dataset_name",
                help="Currently only ShopID10K (amazon) is supported"
            )
            st.session_state.cfg_cluster_index = st.selectbox(
                "Cross-Validation Split",
                [0, 1, 2, 3, 4],
                index=0,
                key="cfg_cluster_index",
                help="Train on 4 splits, test on 1 held-out split"
            )

        with tabs[1]:
            st.subheader("Model Architecture")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.cfg_vision_model = st.text_input(
                    "Vision Model", value="dinov2_vitb14", key="cfg_vision_model"
                )
                st.session_state.cfg_num_id_tokens = st.number_input(
                    "Q-Former Query Tokens", min_value=4, max_value=128, value=32, key="cfg_num_id_tokens"
                )
                st.session_state.cfg_num_vpt_tokens = st.number_input(
                    "Visual Prompt Tokens", min_value=1, max_value=128, value=32, key="cfg_num_vpt_tokens"
                )
            with c2:
                st.session_state.cfg_llm_model = st.text_input(
                    "LLM Model", value="Qwen/Qwen3-0.6B", key="cfg_llm_model"
                )
                st.session_state.cfg_num_icl_samples = st.number_input(
                    "Few-Shot Examples", min_value=8, max_value=256, value=64, key="cfg_num_icl_samples"
                )
                st.session_state.cfg_num_icl_bs = st.number_input(
                    "ICL Batch Size", min_value=1, max_value=16, value=1, key="cfg_num_icl_bs"
                )

            st.subheader("Loss Weights")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.cfg_icl_loss_weight = st.number_input(
                    "ICL Loss Weight", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="cfg_icl_loss_weight"
                )
            with c2:
                st.session_state.cfg_ot_loss_weight = st.number_input(
                    "OT (WPA) Loss Weight", min_value=0.0, max_value=1.0, value=0.01, step=0.001,
                    format="%.3f", key="cfg_ot_loss_weight"
                )

        with tabs[2]:
            st.subheader("Training Hyperparameters")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.cfg_learning_rate = st.number_input(
                    "Learning Rate", min_value=0.0, value=1e-4, step=1e-5, format="%.5f", key="cfg_learning_rate"
                )
                st.session_state.cfg_per_device_train_batch_size = st.number_input(
                    "Batch Size", min_value=1, max_value=1024, value=256, key="cfg_batch_size"
                )
                st.session_state.cfg_max_steps = st.number_input(
                    "Max Steps", min_value=100, max_value=100000, value=5000, key="cfg_max_steps"
                )
            with c2:
                st.session_state.cfg_weight_decay = st.number_input(
                    "Weight Decay", min_value=0.0, value=0.0, step=0.01, key="cfg_weight_decay"
                )
                st.session_state.cfg_lr_scheduler_type = st.selectbox(
                    "LR Scheduler", ["constant", "cosine", "linear", "polynomial"],
                    index=0, key="cfg_lr_scheduler"
                )
                st.session_state.cfg_fp16 = st.checkbox("FP16 Mixed Precision", value=True, key="cfg_fp16")

        with tabs[3]:
            st.subheader("Advanced Settings")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.cfg_gradient_accumulation_steps = st.number_input(
                    "Gradient Accumulation", min_value=1, max_value=16, value=1, key="cfg_grad_accum"
                )
                st.session_state.cfg_save_steps = st.number_input(
                    "Save Every N Steps", min_value=100, max_value=10000, value=1000, key="cfg_save_steps"
                )
                st.session_state.cfg_save_total_limit = st.number_input(
                    "Max Checkpoints", min_value=1, max_value=50, value=10, key="cfg_save_limit"
                )
                st.session_state.cfg_max_grad_norm = st.number_input(
                    "Max Grad Norm", min_value=0.0, value=0.0, key="cfg_max_grad_norm",
                    help="0 = no clipping"
                )
            with c2:
                st.session_state.cfg_eval_steps = st.number_input(
                    "Eval Every N Steps", min_value=1, max_value=10000, value=20, key="cfg_eval_steps"
                )
                st.session_state.cfg_logging_steps = st.number_input(
                    "Log Every N Steps", min_value=1, max_value=1000, value=1, key="cfg_logging_steps"
                )
                st.session_state.cfg_dataloader_num_workers = st.number_input(
                    "DataLoader Workers", min_value=0, max_value=64, value=4, key="cfg_num_workers"
                )
                st.text_input("Output Dir", value="./checkpoints/vicp_exp", key="cfg_output_dir")

    st.markdown("---")
    st.subheader("Generated Command")
    cmd = build_command()
    st.code(cmd, language="bash")

    st.caption("""
    **To train:** Click the 🚀 **Training** tab in the sidebar, then click "Launch Training"  
    **Or copy** the command above and run it directly in terminal
    """)


def get_current_config():
    return {
        "dataset_name": st.session_state.get("cfg_dataset_name", "amazon"),
        "cluster_index": st.session_state.get("cfg_cluster_index", 0),
        "vision_model": st.session_state.get("cfg_vision_model", "dinov2_vitb14"),
        "llm_model": st.session_state.get("cfg_llm_model", "Qwen/Qwen3-0.6B"),
        "num_id_tokens": st.session_state.get("cfg_num_id_tokens", 32),
        "num_vpt_tokens": st.session_state.get("cfg_num_vpt_tokens", 32),
        "num_icl_samples": st.session_state.get("cfg_num_icl_samples", 64),
        "num_icl_bs": st.session_state.get("cfg_num_icl_bs", 1),
        "icl_loss_weight": st.session_state.get("cfg_icl_loss_weight", 1.0),
        "ot_loss_weight": st.session_state.get("cfg_ot_loss_weight", 0.01),
        "learning_rate": st.session_state.get("cfg_learning_rate", 1e-4),
        "weight_decay": st.session_state.get("cfg_weight_decay", 0.0),
        "per_device_train_batch_size": st.session_state.get("cfg_batch_size", 256),
        "max_steps": st.session_state.get("cfg_max_steps", 5000),
        "fp16": st.session_state.get("cfg_fp16", True),
        "lr_scheduler_type": st.session_state.get("cfg_lr_scheduler", "constant"),
        "gradient_accumulation_steps": st.session_state.get("cfg_grad_accum", 1),
        "max_grad_norm": st.session_state.get("cfg_max_grad_norm", 0),
        "save_steps": st.session_state.get("cfg_save_steps", 1000),
        "eval_steps": st.session_state.get("cfg_eval_steps", 20),
        "logging_steps": st.session_state.get("cfg_logging_steps", 1),
        "dataloader_num_workers": st.session_state.get("cfg_num_workers", 4),
        "dataloader_persistent_workers": True,
        "dataloader_pin_memory": False,
        "dataloader_drop_last": True,
        "save_total_limit": st.session_state.get("cfg_save_limit", 10),
        "save_safetensors": False,
        "ddp_find_unused_parameters": False,
        "report_to": "none",
        "output_dir": st.session_state.get("cfg_output_dir", "./checkpoints/vicp_exp"),
    }


def build_command():
    cfg = get_current_config()
    parts = ["python train_vpt_lora.py"]
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v and k != "fp16":
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " \\\n    ".join(parts)


def save_config():
    cfg = get_current_config()
    name = f"exp_cluster{cfg['cluster_index']}_bs{cfg['per_device_train_batch_size']}_lr{cfg['learning_rate']}"
    path = CONFIG_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    st.success(f"Saved to {path}")


def load_saved_config(path):
    with open(path) as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        session_key = config_to_session_key(k)
        if session_key:
            st.session_state[session_key] = v
    st.rerun()


def load_default_config():
    for k, v in {
        "cfg_dataset_name": "amazon", "cfg_cluster_index": 0,
        "cfg_vision_model": "dinov2_vitb14", "cfg_llm_model": "Qwen/Qwen3-0.6B",
        "cfg_num_id_tokens": 32, "cfg_num_vpt_tokens": 32,
        "cfg_num_icl_samples": 64, "cfg_num_icl_bs": 1,
        "cfg_icl_loss_weight": 1.0, "cfg_ot_loss_weight": 0.01,
        "cfg_learning_rate": 1e-4, "cfg_weight_decay": 0.0,
        "cfg_batch_size": 256, "cfg_max_steps": 5000,
        "cfg_fp16": True, "cfg_lr_scheduler": "constant",
        "cfg_grad_accum": 1, "cfg_max_grad_norm": 0,
        "cfg_save_steps": 1000, "cfg_eval_steps": 20,
        "cfg_logging_steps": 1, "cfg_num_workers": 4,
        "cfg_output_dir": "./checkpoints/vicp_exp",
    }.items():
        st.session_state[k] = v
    st.rerun()


def config_to_session_key(k):
    mapping = {
        "dataset_name": "cfg_dataset_name", "cluster_index": "cfg_cluster_index",
        "vision_model": "cfg_vision_model", "llm_model": "cfg_llm_model",
        "num_id_tokens": "cfg_num_id_tokens", "num_vpt_tokens": "cfg_num_vpt_tokens",
        "num_icl_samples": "cfg_num_icl_samples", "num_icl_bs": "cfg_num_icl_bs",
        "icl_loss_weight": "cfg_icl_loss_weight", "ot_loss_weight": "cfg_ot_loss_weight",
        "learning_rate": "cfg_learning_rate", "weight_decay": "cfg_weight_decay",
        "per_device_train_batch_size": "cfg_batch_size", "max_steps": "cfg_max_steps",
        "fp16": "cfg_fp16", "lr_scheduler_type": "cfg_lr_scheduler",
        "gradient_accumulation_steps": "cfg_grad_accum", "max_grad_norm": "cfg_max_grad_norm",
        "save_steps": "cfg_save_steps", "eval_steps": "cfg_eval_steps",
        "logging_steps": "cfg_logging_steps", "dataloader_num_workers": "cfg_num_workers",
        "output_dir": "cfg_output_dir",
    }
    return mapping.get(k, None)
