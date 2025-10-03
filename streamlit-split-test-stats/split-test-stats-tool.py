import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Analysis",
    page_icon="🧪",
    layout="wide"
)

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Welcome!
    **Select a page from the sidebar** that fits your stats use-case!
"""
)

st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

if os.path.exists("pages"):
    st.write("Pages folder exists!")
    st.write("Files in pages folder:", os.listdir("pages"))
else:
    st.write("Pages folder NOT found")
