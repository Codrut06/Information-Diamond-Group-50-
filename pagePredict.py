# pagePredict.py — KNN Model Evaluation and Prediction

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Model — Evaluation and Prediction", layout="wide")
st.title("Match Predictor (KNN)")
st.caption("Test the trained model on existing data or generate new random pairs to evaluate prediction results.")

# ----------------- PATH HELPERS -----------------
def get_search_dirs() -> list[Path]:
    """Return potential directories to search for required files."""
    dirs: list[Path] = [Path.cwd()]
    if sys.path and sys.path[0]:
        try:
            dirs.append(Path(sys.path[0]).resolve())
        except Exception:
            pass
    for p in [Path("./proj_regr"), Path("../proj_regr")]:
        try:
            dirs.append(p.resolve())
        except Exception:
            pass
    seen, uniq = set(), []
    for d in dirs:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


SEARCH_DIRS = get_search_dirs()


def find_file(name: str) -> Path | None:
    """Search for a file in known project directories."""
    for base in SEARCH_DIRS:
        p = base / name
        if p.exists():
            return p
    return None


# ----------------- LOADERS -----------------
@st.cache_resource(show_spinner=False)
def _load_pickle_or_joblib(path: Path):
    """Load a serialized object using pickle or joblib."""
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            from joblib import load as joblib_load
            return joblib_load(path)
        except Exception as e2:
            try:
                with open(path, "rb") as f:
                    head = f.read(16)
            except Exception:
                head = b"<unreadable>"
            raise RuntimeError(
                f"'{path.name}' is not a valid pickle/joblib file. "
                f"First bytes: {head!r}. Please re-save the object from the training script."
            ) from e2


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the scaler used during training."""
    p = find_file("scaler.pkl") or find_file("scaler.joblib")
    if not p:
        raise FileNotFoundError("Missing scaler.pkl or scaler.joblib.")
    return _load_pickle_or_joblib(p)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained KNN model."""
    p = find_file("knn_model.pkl") or find_file("knn_model.joblib")
    if not p:
        raise FileNotFoundError("Missing knn_model.pkl or knn_model.joblib.")
    return _load_pickle_or_joblib(p)


@st.cache_data
def load_cleaned() -> pd.DataFrame:
    """Load the cleaned dataset used for training and evaluation."""
    p = find_file("cleanedData.csv")
    if not p:
        raise FileNotFoundError("Missing cleanedData.csv.")
    return pd.read_csv(p)


@st.cache_data
def load_feature_list() -> List[str]:
    """Load or infer the list of feature columns."""
    p = find_file("feature_columns.txt")
    if p:
        cols = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if cols:
            return cols
    model, scaler = load_model(), load_scaler()
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    raise FileNotFoundError("Missing feature_columns.txt and could not infer features from model or scaler.")


# ----------------- UTILITIES -----------------
TYPO_MAP: Dict[str, str] = {
    "ambitous_o": "ambitious_o",
    "intellicence_important": "intelligence_important",
    "ambtition_important": "ambition_important",
}


def coerce_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Ensure dataframe matches model features and is numeric."""
    cols_fixed = []
    missing_real: List[str] = []
    for c in feature_cols:
        c_real = TYPO_MAP.get(c, c)
        if c_real in df.columns:
            cols_fixed.append(c_real)
        else:
            df[c_real] = 0
            cols_fixed.append(c_real)
            missing_real.append(c)
    if missing_real:
        warnings.warn(f"Created {len(missing_real)} missing feature columns: {missing_real}")
    X = df[cols_fixed].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0)


def get_y_true(df: pd.DataFrame) -> pd.Series:
    """Extract the target column from the dataset."""
    for cand in ["match", "Match", "target", "y"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce").fillna(0).astype(int)
    raise KeyError("Missing target column 'match' in cleanedData.csv.")


# ----------------- MODEL ARTIFACTS -----------------
st.markdown("### Model Artifacts")
st.markdown(
    """
<div style="
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 10px;">
  <b>Required files:</b><br>
  • <code>knn_model.pkl</code> or <code>knn_model.joblib</code><br>
  • <code>scaler.pkl</code> or <code>scaler.joblib</code><br>
  • <code>feature_columns.txt</code> (optional, inferred if missing)<br>
  • <code>cleanedData.csv</code>
</div>
""",
    unsafe_allow_html=True,
)

presence = {
    "Model (pkl/joblib)": bool(find_file("knn_model.pkl") or find_file("knn_model.joblib")),
    "Scaler (pkl/joblib)": bool(find_file("scaler.pkl") or find_file("scaler.joblib")),
    "Feature list (txt)": bool(find_file("feature_columns.txt")),
    "Cleaned data (csv)": bool(find_file("cleanedData.csv")),
}
cols = st.columns(len(presence))
for (label, ok), c in zip(presence.items(), cols):
    c.markdown(f"{'✅' if ok else '❌'} *{label}*")

# Load all artifacts
try:
    scaler = load_scaler()
    model = load_model()
    feature_cols = load_feature_list()
    df_all = load_cleaned()
    y_all = get_y_true(df_all)
    st.success("All model artifacts loaded successfully.")
except Exception as e:
    st.error(f"{e}")
    st.stop()


# ================= RANDOM EVALUATION =================
st.subheader("Evaluate the model on random samples")

n_max = int(min(10000, len(df_all)))
n_tests = st.number_input(
    "Number of samples to test", min_value=1, max_value=n_max, value=min(50, n_max), step=1
)

if "eval_nonce" not in st.session_state:
    st.session_state.eval_nonce = 0

if st.button("Run Evaluation", type="primary"):
    st.session_state.eval_nonce += 1
    random_seed = int(np.random.SeedSequence().generate_state(1)[0])
    sample = df_all.sample(n=n_tests, random_state=random_seed)
    y_true = get_y_true(sample)

    X = coerce_feature_frame(sample, feature_cols)
    try:
        X_scaled = scaler.transform(X.values)
    except Exception:
        X_scaled = X.values

    y_pred = np.asarray(model.predict(X_scaled)).astype(int)

    results = pd.DataFrame(
        {"Row Index": sample.index, "Predicted": y_pred, "Actual": y_true.values}
    )
    results["Correct"] = (results["Predicted"] == results["Actual"]).astype(int)

    st.subheader("Results")
    st.dataframe(results, use_container_width=True)

    total_right = int(results["Correct"].sum())
    total_wrong = int(len(results) - total_right)
    c1, c2 = st.columns(2)
    c1.metric("Correct predictions", f"{total_right}")
    c2.metric("Incorrect predictions", f"{total_wrong}")

    try:
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    except Exception:
        y_t = y_true.to_numpy().astype(int)
        y_p = y_pred.astype(int)
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        cm = np.array([[tn, fp], [fn, tp]])
        acc = float((tp + tn) / max(1, len(y_t)))
        prec = float(tp / max(1, tp + fp))

    st.subheader("Metrics")
    st.markdown("*Confusion Matrix (rows = actual, columns = predicted)*")
    st.table(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision (class=1)", f"{prec:.3f}")

    st.caption(
        "The table includes the dataset index, predicted match (0/1), and actual outcome."
    )


# ================= SYNTHETIC PAIR GENERATION =================
st.markdown("---")
st.subheader("Generate synthetic pairs and predict outcomes")

col1, col2 = st.columns(2)
with col1:
    n_gen = st.number_input(
        "Number of pairs to generate", min_value=1, max_value=2000, value=20, step=1
    )
with col2:
    replace_sampling = st.checkbox(
        "Allow sampling with replacement (greater variation)", value=True
    )

if st.button("Generate and Predict"):
    ref = coerce_feature_frame(df_all, feature_cols)
    synth = ref.sample(
        n=int(n_gen), replace=bool(replace_sampling), random_state=None
    ).reset_index(drop=True)

    try:
        X_new = scaler.transform(synth.values)
    except Exception:
        X_new = synth.values

    y_new = np.asarray(model.predict(X_new)).astype(int)

    detailed = synth.copy()
    detailed.insert(0, "Date", np.arange(1, len(detailed) + 1, dtype=int))
    detailed["Predicted Match"] = y_new
    st.markdown("**Generated Pairs (features + predicted outcome)**")
    st.dataframe(detailed, use_container_width=True)

    summary = pd.DataFrame(
        {"Date": np.arange(1, len(y_new) + 1, dtype=int), "Predicted Match": y_new}
    )
    st.markdown("**Summary: Pair ID and predicted match**")
    st.dataframe(summary, use_container_width=True)
