import streamlit as st

st.title("VICP Training Platform")
st.write("If you see this, Streamlit is working correctly.")

page = st.sidebar.radio("Page", ["Overview", "Configuration", "Training", "Results", "Data Browser"])

if page == "Overview":
    st.header("Overview")
    st.markdown("""
## VICP: Visual In-Context Prompting  
**ICCV 2025** · Generalizable Object ReID

### Architecture
1. **DINOv2 (frozen + LoRA)** extracts image features
2. **Qwen3-0.6B (frozen)** infers identity rules from few-shot examples
3. **Visual Prompts** from LLM guide DINOv2 for task-specific feature extraction

### Key Components
| Component | Details |
|-----------|---------|
| Vision Backbone | DINOv2 ViT-B/14 (frozen, LoRA on last 4 blocks) |
| LLM | Qwen3-0.6B (frozen, ICL reasoning) |
| Connector | Q-Former + Prompt MLP (trainable) |
| Losses | Hard Triplet + ICL Cross-Entropy + OT (WPA) |
| Dataset | ShopID10K (34 categories) |
    """)

elif page == "Configuration":
    st.header("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Dataset", ["amazon"], key="ds")
        st.selectbox("CV Split", [0, 1, 2, 3, 4], key="split")
        st.text_input("Vision Model", "dinov2_vitb14", key="vis")
    with col2:
        st.text_input("LLM", "Qwen/Qwen3-0.6B", key="llm")
        st.number_input("Batch Size", 1, 1024, 256, key="bs")
        st.number_input("Max Steps", 100, 100000, 5000, key="steps")

    st.subheader("Generated Command")
    st.code(f"""python train_vpt_lora.py \\
    --dataset_name {st.session_state.get("ds", "amazon")} \\
    --cluster_index {st.session_state.get("split", 0)} \\
    --vision_model {st.session_state.get("vis", "dinov2_vitb14")} \\
    --llm_model {st.session_state.get("llm", "Qwen/Qwen3-0.6B")} \\
    --per_device_train_batch_size {st.session_state.get("bs", 256)} \\
    --max_steps {st.session_state.get("steps", 5000)} \\
    --fp16 True \\
    --output_dir ./checkpoints/vicp_exp""", language="bash")

elif page == "Training":
    st.header("Training")
    st.info("Click below to launch training as a terminal command:")
    st.code("""
# Run this in your terminal:
python train_vpt_lora.py \\
    --dataset_name amazon \\
    --cluster_index 0 \\
    --per_device_train_batch_size 256 \\
    --max_steps 5000 \\
    --fp16 True \\
    --learning_rate 1e-4 \\
    --output_dir ./checkpoints/vicp_exp
""", language="bash")

elif page == "Results":
    st.header("Results")
    st.info("Results show here after training completes. Metrics: Rank-1, Rank-5, mAP")

elif page == "Data Browser":
    st.header("Data Browser")
    from pathlib import Path
    root = st.text_input("Dataset Root", "../groundingdino_cropped")
    if not Path(root).exists():
        st.warning(f"Dataset not found at {root}")
        st.markdown("Download: [Google Drive](https://drive.google.com/drive/folders/1ubm0oo8-5wXLocoHIk5yt1CtzgXTg_1h)")

st.sidebar.markdown("---")
st.sidebar.caption("Prompt: learn from few-shot examples on seen categories, generalize to unseen ones.")
