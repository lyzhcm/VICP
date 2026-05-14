""" Results Page - Evaluation Visualization """
import streamlit as st
import json
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    st.title("Evaluation Results")

    result_dir = st.text_input("Checkpoint/Results Directory", value="./checkpoints/vicp_exp")

    cols = st.columns(4)
    with cols[0]:
        load_btn = st.button("📂 Load Results", use_container_width=True)
    with cols[1]:
        demo_btn = st.button("🎲 Show Demo Results", use_container_width=True)

    if load_btn:
        st.info("Evaluation runs automatically every `--eval_steps` during training.")
        st.info("Results are logged and can be viewed from WandB or the training log.")

    if demo_btn:
        show_demo_results()

    st.markdown("---")

    st.subheader("Expected Performance (from paper)")
    render_expected_performance()

    st.markdown("---")

    st.subheader("Per-Category Performance")
    render_per_category_demo()

    st.markdown("---")

    st.subheader("Upload Custom Results")
    uploaded = st.file_uploader("Upload results JSON or CSV", type=["json", "csv"])
    if uploaded:
        st.json(json.load(uploaded))


def show_demo_results():
    st.success("Demo results loaded!")

    categories = [
        "tackle_box", "portable_chair", "bicycle", "poster_tube", "duffel_bag",
        "sports_equipment", "hat", "box", "purse", "hardshell_case", "suitcase",
        "umbrella", "musical_instrument", "trash_can", "hand_truck", "pet_carrier",
        "sports_ball", "mobile_phone", "portable_speaker", "binoculars", "headphones",
        "beverage_bottle", "tire_wheel", "jacket", "cooler", "book", "stroller",
        "backpack", "food_container", "cart", "bucket", "shoes", "bicycle_helmet", "skateboard"
    ]

    import numpy as np
    np.random.seed(42)

    splits = [
        categories[0:7], categories[7:14], categories[14:21],
        categories[21:28], categories[28:34]
    ]

    v1, v2, v3 = st.columns(3)
    v1.metric("Mean Rank-1", "82.5%")
    v2.metric("Mean Rank-5", "94.3%")
    v3.metric("Mean mAP", "78.1%")


def render_expected_performance():
    import numpy as np

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ReID (Rank-1, Rank-5, mAP)", "Verification (AUC, Accuracy)"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    methods = ["VICP (Ours)", "DINOv2 Zero-shot", "CLIP Zero-shot"]
    reid_metrics = {
        "Rank-1": [83.1, 37.8, 28.4],
        "Rank-5": [94.5, 56.2, 47.1],
        "mAP": [78.6, 28.1, 19.2],
    }

    x = np.arange(len(methods))
    width = 0.25
    colors = ["#636EFA", "#00CC96", "#EF553B"]

    for i, (name, vals) in enumerate(reid_metrics.items()):
        fig.add_trace(go.Bar(
            x=methods, y=vals, name=name,
            text=[f"{v}%" for v in vals], textposition="outside",
            marker_color=colors[i]
        ), row=1, col=1)

    verif_metrics = {
        "AUC": [92.1, 72.4, 68.5],
        "Accuracy": [84.3, 65.2, 60.8],
    }
    vcolors = ["#636EFA", "#EF553B"]
    for i, (name, vals) in enumerate(verif_metrics.items()):
        fig.add_trace(go.Bar(
            x=methods, y=vals, name=name,
            text=[f"{v}%" for v in vals], textposition="outside",
            marker_color=vcolors[i]
        ), row=1, col=2)

    fig.update_layout(
        height=450, showlegend=False, template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def render_per_category_demo():
    import numpy as np
    np.random.seed(42)

    categories = [
        "bicycle", "hat", "backpack", "headphones", "sports_ball",
        "shoes", "suitcase", "jacket", "book", "umbrella",
        "mobile_phone", "cooler", "purse", "tire_wheel", "sports_equipment",
        "binoculars", "beverage_bottle", "cart", "portable_chair", "bucket",
        "hand_truck", "trash_can", "bicycle_helmet", "food_container",
        "musical_instrument", "portable_speaker", "pet_carrier", "stroller",
        "tackle_box", "hardshell_case", "poster_tube", "skateboard", "duffel_bag", "box",
    ]

    base_rank1 = np.random.uniform(70, 90, len(categories)).tolist()
    base_rank5 = np.random.uniform(85, 98, len(categories)).tolist()
    base_map = np.random.uniform(60, 85, len(categories)).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories, x=base_rank1, name="Rank-1",
        orientation="h", marker_color="#636EFA",
        text=[f"{v:.1f}%" for v in base_rank1], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=categories, x=base_rank5, name="Rank-5",
        orientation="h", marker_color="#00CC96",
        text=[f"{v:.1f}%" for v in base_rank5], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=categories, x=base_map, name="mAP",
        orientation="h", marker_color="#EF553B",
        text=[f"{v:.1f}%" for v in base_map], textposition="outside",
    ))

    fig.update_layout(
        height=750, barmode="group", template="plotly_white",
        margin=dict(l=10, r=80, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, 100], title="Score (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Demo data — actual results will appear here after evaluation")
