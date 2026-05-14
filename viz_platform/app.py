""" VICP Visual Training Platform - Main Entry Point """
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="VICP Training Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pages import overview, config, training, results, data_browser

PAGES = {
    "🏠 Overview": overview,
    "⚙️ Configuration": config,
    "🚀 Training": training,
    "📊 Results": results,
    "📁 Data Browser": data_browser,
}

st.sidebar.title("🔬 VICP Platform")
st.sidebar.markdown("---")
st.sidebar.markdown("ICCV 2025 · Generalizable Object ReID")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page].render()
