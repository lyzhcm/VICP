import streamlit as st

st.set_page_config(page_title="Test", layout="wide")
st.title("Hello VICP!")
st.write("This is a minimal test. If you see this, Streamlit is working.")
st.sidebar.text("Sidebar test")

import sys
st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)
