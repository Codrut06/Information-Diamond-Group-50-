# pageHome.py — Landing page (overview)
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Overview — Speed Dating App", layout="wide")

# ------------------------------ NAVBAR ---------------------------------- #
def navbar() -> None:
    st.markdown(
        """
        <style>
        .nav-wrap {
            position: sticky; top: 0; z-index: 999;
            backdrop-filter: blur(8px);
            background: rgba(0,0,0,0.05);
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding: 8px 12px; border-radius: 12px;
        }
        .pill {
            display:inline-block; margin-right: .5rem; padding: .4rem .8rem;
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 999px; text-decoration: none;
            font-size: .95rem;
        }
        .pill:hover { border-color: rgba(33,196,184,.9); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nav-wrap">
          <a class="pill" href="#overview">Overview</a>
          <a class="pill" href="#how-to-use">How to use</a>
          <a class="pill" href="#performance">Performance</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------ SECTIONS -------------------------------- #
def section_overview() -> None:
    st.header("Overview", anchor="overview")
    st.write("")
    st.markdown(
        """
        <div style="
            background: linear-gradient(125deg, #00008B 0%, #ADD8E6 100%);
            padding: 28px 20px; border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.15); color: white;">
          This project enables interactive exploration of a speed dating dataset:
          querying, filtering, and analyzing behaviors and outcomes. It also
          includes a classification model to estimate match outcomes for single
          dates or batches based on participant inputs.
          <br><br>
          The app blends data exploration, visual analytics, and ML to examine
          what drives connection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <style>
        .equal-box {
            background: #527FC3; padding: 20px; border-radius: 12px;
            height: 200px; display: flex; align-items: center;
            justify-content: center; text-align: center; box-sizing: border-box;
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='equal-box'><h4>Which traits matter most when people connect?</h4></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='equal-box'><h4>Shared interests vs. appearance — which influences more?</h4></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='equal-box'><h4>Can we predict attraction from traits and preferences alone?</h4></div>", unsafe_allow_html=True)


def section_how_to() -> None:
    st.header("How to use", anchor="how-to-use")
    st.write("")
    st.markdown(
        """
        <div style="
            background: linear-gradient(140deg, #00008B 0%, #92CBDD 100%);
            padding: 20px; border-radius: 12px; color: #fff;">
          <h5>1) Explore the Data</h5>
          Use the <b>Explore</b> page to inspect demographics, match ratios, and interest distributions.

          <h5>2) Visualize Relationships</h5>
          The <b>Explore</b> and <b>SQL Lab</b> pages show how age, race, or attractiveness relate to matches.

          <h5>3) Run SQL Queries</h5>
          The <b>SQL Lab</b> lets you write SELECT queries (pandasql) to analyze patterns directly.

          <h5>4) Predict a Match</h5>
          The <b>Model — Train & Predict</b> page simulates dates or tests the trained KNN model.
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_performance() -> None:
    st.header("Performance", anchor="performance")
    st.write("")
    st.markdown(
        """
        <style>
        .equal-height {
            height: 330px; display: flex; flex-direction: column;
            justify-content: center; box-sizing: border-box; color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c4, c5 = st.columns([2, 1], gap="large")
    with c4:
        st.markdown(
            """
            <div class="equal-height"
                 style="background: linear-gradient(135deg, #00008B 0%, #ADD8E6 100%);
                        padding: 20px; border-radius: 12px;">
              <div style="background: rgba(0, 123, 255, 0.25);
                          padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                Large dataset: ~8,000+ rows, 120+ columns
              </div>
              <div style="background: rgba(0, 123, 255, 0.25);
                          padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                Classification focus: reports Accuracy / Precision / Confusion Matrix
              </div>
              <div style="background: rgba(0, 123, 255, 0.25);
                          padding: 20px; border-radius: 12px;">
                Interactive: run your own queries and review results quickly
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="equal-height"
                 style="background: #527FC3; padding: 20px; border-radius: 12px;">
              <h5>Scalability</h5>
              Deploy on Streamlit Cloud or servers.
              <br><br>
              Modular design supports new datasets and retraining
              without breaking the interface.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Feature importance (illustrative)")
    st.caption("Example values — replace with model-specific scores.")
    data = {
        "Feature": [
            "Self-rated features",
            "Shared interests",
            # "Fits stated preferences",
            "Other-rated features",
            "Demographics",
        ],
        "MAE": [0.169, 0.187, 0.152, 0.194],
    }
    df_imp = pd.DataFrame(data).sort_values("MAE", ascending=True)
    st.bar_chart(df_imp, x="Feature", y="MAE", height=380, use_container_width=True)


# ------------------------------- RENDER --------------------------------- #
def main() -> None:
    navbar()
    st.write("")
    section_overview()
    st.write("")
    section_how_to()
    st.write("")
    section_performance()


# Always call directly (since router doesn’t set __name__)
main()
