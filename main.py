# main.py — single-app router (without /pages)
import os
import io
import streamlit as st

st.set_page_config(page_title="Speed Dating — App", layout="wide")

st.title("Speed Dating — Full App")
st.caption("Select a page from the sidebar.")

# ----------------- PAGE ROUTER -----------------
PAGES = {
    "Home": "pageHome.py",
    "Explore (Diagrams)": "pageDiagrams.py",
    "SQL Lab": "pageQuerry.py",
    "Model — Train & Predict": "pagePredict.py",
}

choice = st.sidebar.radio("Pages", list(PAGES.keys()), index=0)

# ----------------- SCRIPT LOADER -----------------
def run_streamlit_script(filepath: str):
    """Executes one of the selected Streamlit page scripts."""
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return
    with io.open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Prevent Streamlit from throwing “set_page_config can only be called once”
    orig_set = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        # Set __name__ properly so 'if __name__ == "__main__":' works
        ns = {"__name__": "__main__"}
        exec(compile(code, filepath, "exec"), ns, ns)
    finally:
        st.set_page_config = orig_set


# ----------------- RUN SELECTED PAGE -----------------
run_streamlit_script(PAGES[choice])
