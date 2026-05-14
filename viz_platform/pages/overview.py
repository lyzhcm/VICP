""" Overview Page - Architecture Visualization """
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def render():
    st.title("VICP: Visual In-Context Prompting")
    st.markdown(
        """
    **ICCV 2025** · Zhizhong Huang, Xiaoming Liu  
    *Generalizable Object Re-Identification via Visual In-Context Prompting*
    """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Core Idea")
        st.markdown(
            """
        Traditional ReID methods require dataset-specific training for each object 
        category (person, vehicle, product, etc.). **VICP** introduces a paradigm shift:

        1. **LLM learns semantic identity rules** from few-shot positive/negative image pairs
        2. **LLM generates visual prompts** that guide a Vision Foundation Model (DINOv2)
        3. **Direct generalization** to unseen categories without fine-tuning

        > Train on 27 categories → evaluate on 7 held-out categories, zero-shot.
        """
        )

    with col2:
        st.subheader("Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vision Backbone", "DINOv2\nViT-B/14")
        m2.metric("LLM", "Qwen3\n0.6B")
        m3.metric("Trainable\nParams", "~25M")
        m4.metric("Dataset\nSize", "ShopID10K")

    st.markdown("---")

    st.subheader("Architecture Pipeline")
    render_architecture_viz()

    st.markdown("---")

    st.subheader("Components Detail")
    tabs = st.tabs(["Stage 1: ICL", "Stage 2: Visual Prompting", "Stage 3: ReID", "Loss Functions"])

    with tabs[0]:
        st.markdown(
            """
        ### Stage 1: In-Context Learning (ICL)
        
        **Input:** Few-shot image pairs (same/different identity) from the target category
        
        1. Pairs are fed through a **frozen DINOv2** (encoder_copy) to extract [CLS] features
        2. Features from pairs are concatenated and shown to the **frozen Qwen3-0.6B LLM**
        3. LLM is prompted with [YES]/[NO] tokens to predict if pairs belong to the same identity
        4. The LLM's **hidden states** capture rich semantic understanding of the category's identity
        
        **ICL Loss:** Cross-entropy on YES/NO predictions — teaches LLM to understand identity similarity
        """
        )

    with tabs[1]:
        st.markdown(
            """
        ### Stage 2: Visual Prompt Generation
        
        1. **Learnable Query Embeddings** (num_vpt_tokens × num_layers) are appended to LLM input
        2. LLM forward pass generates hidden states for these query tokens
        3. **Prompt MLP** projects hidden states from LLM space → DINOv2 embedding space
        4. Result: **32 visual prompt tokens** per transformer layer
        
        These prompts encode the LLM's semantic understanding of "what makes two objects the same identity"
        """
        )

    with tabs[2]:
        st.markdown(
            """
        ### Stage 3: Re-Identification
        
        1. Visual prompts are prepended to DINOv2's transformer layers (last N layers)
        2. DINOv2 with **LoRA adapters** (rank=128 on QKV of last 4 blocks) processes images
        3. The prompts act as "task instructions" steering feature extraction toward identity-discriminative patterns
        4. Final [CLS] token output is L2-normalized and used for similarity search
        
        **Key insight:** Same DINOv2 weights, but prompts from different categories produce different feature spaces!
        """
        )

    with tabs[3]:
        st.markdown(
            """
        ### Three Combined Losses:
        
        | Loss | Weight | Purpose |
        |------|--------|---------|
        | **ID Loss** (Hard Triplet) | 1.0 | Pull same-ID features together, push different-ID apart |
        | **ICL Loss** (Cross-Entropy) | 1.0 | Teach LLM to understand identity semantics from few examples |
        | **OT Loss** (Wasserstein Patch Alignment) | 0.01 | Align patch-level features between positive pairs |
        
        **Feature Collapse Monitor:** `std` metric — tracks if features collapse to a single point (important for ReID)
        """
        )

    st.markdown("---")
    st.info(
        "**📖 Paper:** [arXiv:2508.21222](https://arxiv.org/abs/2508.21222)  "
        "**💾 Code:** `models.py` (core model), `train_vpt_lora.py` (training entry), "
        "`ops/` (LoRA, losses, evaluation, dataset)"
    )


def render_architecture_viz():
    fig = go.Figure()

    nodes = [
        {"id": 1, "name": "Image Pairs", "group": "input", "x": 0, "y": 0.5},
        {"id": 2, "name": "DINOv2\n(frozen)", "group": "encoder", "x": 0.2, "y": 0.5},
        {"id": 3, "name": "Q-Former", "group": "connector", "x": 0.37, "y": 0.5},
        {"id": 4, "name": "Qwen3-0.6B\n(frozen LLM)", "group": "llm", "x": 0.55, "y": 0.7},
        {"id": 5, "name": "Prompt MLP", "group": "connector", "x": 0.72, "y": 0.5},
        {"id": 6, "name": "Visual\nPrompts", "group": "prompt", "x": 0.82, "y": 0.3},
        {"id": 7, "name": "DINOv2\n+LoRA", "group": "encoder", "x": 0.82, "y": 0.65},
        {"id": 8, "name": "ReID\nFeatures", "group": "output", "x": 1.0, "y": 0.65},
        {"id": 9, "name": "ICL Loss\n(LLM)", "group": "loss", "x": 0.55, "y": 0.85},
        {"id": 10, "name": "ID Loss\n(Triplet)", "group": "loss", "x": 1.0, "y": 0.8},
        {"id": 11, "name": "OT Loss\n(WPA)", "group": "loss", "x": 1.0, "y": 0.5},
    ]

    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (4, 9), (8, 10), (7, 11), (2, 4),
    ]

    colors = {
        "input": "#636EFA", "encoder": "#00CC96", "connector": "#AB63FA",
        "llm": "#FFA15A", "prompt": "#19D3F3", "output": "#FF6692", "loss": "#EF553B"
    }

    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode="markers+text",
            marker=dict(size=45, color=colors[node["group"]], line=dict(width=2, color="white")),
            text=node["name"],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo="text",
            name=node["name"],
            showlegend=False,
        ))

    for s, t in edges:
        sn = nodes[s - 1]
        tn = nodes[t - 1]
        fig.add_trace(go.Scatter(
            x=[sn["x"], tn["x"]],
            y=[sn["y"], tn["y"]],
            mode="lines",
            line=dict(width=2, color="rgba(150,150,150,0.6)"),
            hoverinfo="none",
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.05]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
