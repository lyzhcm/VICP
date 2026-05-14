""" Data Browser Page - Explore ShopID10K Dataset """
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
import pandas as pd
import plotly.express as px


def render():
    st.title("Dataset Browser")

    root_dir = st.text_input("Dataset Root", value="./groundingdino_cropped")

    if not Path(root_dir).exists():
        st.warning(f"📁 Dataset not found at `{root_dir}`")
        st.markdown("""
        ### How to get the dataset:
        
        1. **Download ShopID10K** from [Google Drive](https://drive.google.com/drive/folders/1ubm0oo8-5wXLocoHIk5yt1CtzgXTg_1h)
        2. Extract to `./groundingdino_cropped/`
        3. Expected structure:
        ```
        groundingdino_cropped/
        ├── images/          # All cropped images
        ├── split/
        │   ├── bicycle/
        │   │   ├── train.csv
        │   │   ├── test.txt
        │   │   └── verification.csv
        │   ├── backpack/
        │   ├── ...
        ```
        """)
        return

    try:
        from config import data_config
        all_categories = data_config.get("amazon", {}).get("classes", [])
        splits = data_config.get("amazon", {}).get("splits", [])
    except ImportError:
        st.error("Cannot import config.py. Make sure you're in the VICP project root.")
        return

    tabs = st.tabs(["📊 Category Overview", "🖼️ Image Browser", "📋 Split Details", "📈 Statistics"])

    with tabs[0]:
        st.subheader(f"All {len(all_categories)} Categories")
        render_category_pie(all_categories, splits)
        render_category_grid(all_categories, splits)

    with tabs[1]:
        st.subheader("Image Browser")
        render_image_browser(root_dir, all_categories)

    with tabs[2]:
        st.subheader("Cross-Validation Splits")
        render_split_details(root_dir, splits)

    with tabs[3]:
        st.subheader("Dataset Statistics")
        render_statistics(root_dir, all_categories)


def render_category_pie(categories, splits):
    split_labels = [f"Split {i}" for i in range(len(splits))]
    source_data = []
    for i, split in enumerate(splits):
        for cat in split:
            source_data.append({"Category": cat, "Split": split_labels[i]})

    if not source_data:
        return
    df = pd.DataFrame(source_data)
    fig = px.sunburst(
        df, path=["Split", "Category"],
        title="Category Distribution Across Splits",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_category_grid(categories, splits):
    st.subheader("Categories by Split")
    for i, split in enumerate(splits):
        st.markdown(f"**Split {i}:** {', '.join(split)}")


def render_image_browser(root_dir, categories):
    selected_cat = st.selectbox("Select Category", categories)

    train_csv = Path(root_dir) / "split" / selected_cat / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        st.markdown(f"**{len(df)} training images** for `{selected_cat}`")

        n_cols = 4
        n_samples = min(12, len(df))

        for i in range(0, n_samples, n_cols):
            cols = st.columns(n_cols)
            for j in range(n_cols):
                idx = i + j
                if idx < n_samples:
                    img_path = os.path.join(root_dir, "images", df.iloc[idx]["filename"])
                    with cols[j]:
                        if os.path.exists(img_path):
                            try:
                                from PIL import Image
                                img = Image.open(img_path).convert("RGB")
                                st.image(img, use_container_width=True, caption=f"#{idx}")
                            except Exception:
                                st.warning(f"Cannot load image {idx}")
                        else:
                            st.warning(f"Image not found: {os.path.basename(img_path)}")
    else:
        st.warning(f"No train.csv found for `{selected_cat}`")

    st.markdown("---")
    st.subheader("Test Data")
    test_txt = Path(root_dir) / "split" / selected_cat / "test.txt"
    if test_txt.exists():
        with open(test_txt) as f:
            lines = f.readlines()
        st.markdown(f"**{len(lines)} test images** for ReID evaluation")

        n_cols = 4
        n_samples = min(12, len(lines))
        for i in range(0, n_samples, n_cols):
            cols = st.columns(n_cols)
            for j in range(n_cols):
                idx = i + j
                if idx < n_samples:
                    img_path = os.path.join(root_dir, "images", lines[idx].strip())
                    with cols[j]:
                        if os.path.exists(img_path):
                            try:
                                from PIL import Image
                                img = Image.open(img_path).convert("RGB")
                                st.image(img, use_container_width=True)
                            except Exception:
                                st.warning(f"Cannot load image {idx}")


def render_split_details(root_dir, splits):
    for i, split in enumerate(splits):
        with st.expander(f"Split {i} - {len(split)} categories"):
            total_train, total_test = 0, 0
            rows = []
            for cat in split:
                train_csv = Path(root_dir) / "split" / cat / "train.csv"
                test_txt = Path(root_dir) / "split" / cat / "test.txt"
                n_train = len(pd.read_csv(train_csv)) if train_csv.exists() else 0
                n_test = 0
                if test_txt.exists():
                    with open(test_txt) as f:
                        n_test = len(f.readlines())
                total_train += n_train
                total_test += n_test
                rows.append({"Category": cat, "Train Images": n_train, "Test Images": n_test})

            rows.append({"Category": "**TOTAL**", "Train Images": total_train, "Test Images": total_test})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_statistics(root_dir, categories):
    stats = []
    for cat in categories:
        train_csv = Path(root_dir) / "split" / cat / "train.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            identities = set()
            for f in df["filename"]:
                identities.add(f.split("/")[-2])
            stats.append({
                "Category": cat,
                "Images": len(df),
                "Identities": len(identities),
                "Imgs/ID": round(len(df) / max(len(identities), 1), 1),
            })

    if stats:
        df = pd.DataFrame(stats)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(
            df.sort_values("Images"),
            y="Category", x="Images", orientation="h",
            title="Images per Category",
            color="Images", color_continuous_scale="Viridis",
        )
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
