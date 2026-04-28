import os
import sys
import warnings
import posixpath
import tempfile
import tarfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
from joblib import load

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import shap


# ─────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Reference dataset — must contain exactly the columns the model was trained on
# (the X_train you saved out of the notebook after cells 22-29).
file_path = os.path.join(project_root, "Portfolio/X_train.csv")
dataset   = pd.read_csv(file_path)
dataset   = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]


# ─────────────────────────────────────────────────────────────────────
# AWS credentials (Streamlit secrets)
# ─────────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )


session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ─────────────────────────────────────────────────────────────────────
# Model / feature config
# ─────────────────────────────────────────────────────────────────────
# Only a handful of features are exposed in the UI; every other column
# falls back to the first row of the reference dataset so the endpoint
# always receives the full feature set the model expects.
MODEL_INFO = {
    "endpoint"       : aws_endpoint,
    "explainer"      : "explainer_pca.shap",
    "model_archive"  : "finalized_loan_model.tar.gz",
    "model_joblib"   : "finalized_loan_model.joblib",
    "model_s3_prefix": "sklearn-pipeline-deployment",
    "inputs": [
        # name, min, max, default, step  — ranges chosen for raw IEEE-CIS values
        {"name": "TransactionAmt", "min":   0.0, "max": 10000.0, "default": 100.0, "step":   1.0},
        {"name": "card3",          "min":   0.0, "max":   500.0, "default": 150.0, "step":   1.0},
        {"name": "card6",          "min":   0.0, "max":     5.0, "default":   2.0, "step":   1.0},  # label-encoded
        {"name": "C12",            "min":   0.0, "max":   100.0, "default":   0.0, "step":   1.0},
    ],
}


# ─────────────────────────────────────────────────────────────────────
# S3 helpers (cached so the page doesn't redownload on every interaction)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket, key):
    """Download and return the trained model from S3."""
    s3_client = _session.client("s3")
    archive   = MODEL_INFO["model_archive"]
    local_arc = os.path.join(tempfile.gettempdir(), archive)

    s3_client.download_file(
        Filename=local_arc,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(archive)}",
    )

    extract_dir = tempfile.mkdtemp()
    with tarfile.open(local_arc, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_name = next(f for f in tar.getnames() if f.endswith(".joblib"))

    return joblib.load(os.path.join(extract_dir, joblib_name))


@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):
    """Download and return the SHAP explainer (cached after first load)."""
    s3_client = _session.client("s3")
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)


# ─────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    """Send a single-row DataFrame to the SageMaker endpoint."""
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    # The inference handler accepts a list of records; that keeps the
    # column names attached on the server side.
    payload = input_df.to_dict(orient="records")

    try:
        result = predictor.predict(payload)
        # result is {"prediction": [0/1, ...], "fraud_probability": [float, ...]}
        label_id = int(result["prediction"][0])
        proba    = float(result["fraud_probability"][0])
        mapping  = {0: "Legitimate", 1: "Fraud"}
        return {"label": mapping.get(label_id, str(label_id)), "proba": proba}, 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ─────────────────────────────────────────────────────────────────────
# SHAP explanation
# ─────────────────────────────────────────────────────────────────────
def display_explanation(input_df):
    """Render a SHAP waterfall plot for the fraud (class 1) prediction."""
    try:
        explainer = load_shap_explainer(
            session,
            aws_bucket,
            posixpath.join("explainer", MODEL_INFO["explainer"]),
            os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"]),
        )

        shap_values = explainer(input_df)

        st.subheader("🔍 Decision Transparency (SHAP)")
        fig, _ = plt.subplots(figsize=(10, 4))

        # For binary classifiers SHAP returns shape (n_samples, n_features, n_classes).
        # We slice to row 0, all features, class 1 (fraud).
        sv = shap_values[0]
        if hasattr(sv, "values") and sv.values.ndim == 2:
            sv = sv[:, 1]
        shap.plots.waterfall(sv, show=False)
        st.pyplot(fig)

        # Highlight the single biggest driver
        top_feature = (
            pd.Series(sv.values, index=sv.feature_names).abs().idxmax()
        )
        st.info(
            f"**Business Insight:** The most influential factor in this "
            f"decision was **{top_feature}**."
        )
    except Exception as e:
        st.warning(f"Could not render SHAP explanation: {e}")


# ─────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection — ML Deployment", layout="wide")
st.title("💳 Fraud Detection — ML Deployment")
st.caption(
    "Tuned tree-based classifier (Random Forest / Gradient Boosting) "
    "trained on the IEEE-CIS transaction dataset. Endpoint hosted on "
    "Amazon SageMaker."
)

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    cols         = st.columns(2)
    user_inputs  = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # Build the full feature row: start from the first row of X_train,
    # then overwrite the four user-controlled columns.
    row = dataset.iloc[0:1].copy()
    for name, val in user_inputs.items():
        if name in row.columns:
            row[name] = val
        else:
            st.warning(
                f"Feature `{name}` was not found in X_train.csv — "
                "ignoring it. Check the inputs config."
            )

    res, status = call_model_api(row)

    if status == 200:
        c1, c2 = st.columns(2)
        c1.metric("Prediction", res["label"])
        c2.metric("Fraud Probability", f"{res['proba']:.2%}")

        if res["label"] == "Fraud":
            st.error(
                "⚠️ Flagged as fraudulent. Recommend manual review before "
                "authorization."
            )
        else:
            st.success("✅ Looks legitimate.")

        display_explanation(row)
    else:
        st.error(res)
