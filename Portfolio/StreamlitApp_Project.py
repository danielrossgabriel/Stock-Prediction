import os
import sys
import warnings
import posixpath
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)


# ─────────────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — ML Deployment",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────────────
# AWS credentials from Streamlit secrets
# ─────────────────────────────────────────────────────────────────────
try:
    aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
    aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
    aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
except Exception as e:
    st.error(
        "AWS credentials are missing from Streamlit secrets. "
        "Add your [aws_credentials] block to .streamlit/secrets.toml "
        "or Streamlit Cloud secrets."
    )
    st.stop()


@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )


session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ─────────────────────────────────────────────────────────────────────
# Reference dataset
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_reference_dataset(bucket):
    """
    Load X_train.csv, which gives the app the exact feature columns
    expected by the SageMaker model.

    Order:
    1. Try Portfolio/X_train.csv from GitHub repo.
    2. If missing, download reference/X_train.csv from S3.
    """

    local_path = os.path.join(project_root, "Portfolio", "X_train.csv")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        source = "local GitHub file"
    else:
        temp_path = os.path.join(tempfile.gettempdir(), "X_train.csv")

        try:
            s3_client = session.client("s3")
            s3_client.download_file(
                Bucket=bucket,
                Key="reference/X_train.csv",
                Filename=temp_path
            )

            df = pd.read_csv(temp_path)
            source = "S3 reference/X_train.csv"

        except Exception as e:
            st.error(
                "Could not find X_train.csv locally or in S3.\n\n"
                "Fix this by either:\n"
                "1. Adding Portfolio/X_train.csv to your GitHub repo, or\n"
                "2. Uploading X_train.csv to S3 at reference/X_train.csv.\n\n"
                f"Details: {e}"
            )
            st.stop()

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if df.empty:
        st.error("X_train.csv loaded, but it is empty.")
        st.stop()

    st.sidebar.success(f"Reference data loaded from: {source}")

    return df


dataset = load_reference_dataset(aws_bucket)


# ─────────────────────────────────────────────────────────────────────
# Model / feature config
# ─────────────────────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_pca.shap",
    "model_s3_prefix": "sklearn-pipeline-deployment",
    "inputs": [
        {
            "name": "TransactionAmt",
            "min": 0.0,
            "max": 10000.0,
            "default": 100.0,
            "step": 1.0,
        },
        {
            "name": "card3",
            "min": 0.0,
            "max": 500.0,
            "default": 150.0,
            "step": 1.0,
        },
        {
            "name": "card6",
            "min": 0.0,
            "max": 5.0,
            "default": 2.0,
            "step": 1.0,
        },
        {
            "name": "C12",
            "min": 0.0,
            "max": 100.0,
            "default": 0.0,
            "step": 1.0,
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────
# SHAP helper
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_shap_explainer(bucket, key, local_path):
    """
    Download and return the SHAP explainer from S3.
    Cached so it does not redownload every time the user clicks.
    """

    s3_client = session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key
        )

    with open(local_path, "rb") as f:
        return load(f)


# ─────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    """
    Send one row of feature data to the SageMaker endpoint.
    """

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    # Safer than input_df.to_dict because it converts NumPy/Pandas types
    # into plain JSON-compatible Python values.
    payload = input_df.to_json(orient="records")
    payload = pd.read_json(payload).to_dict(orient="records")

    try:
        result = predictor.predict(payload)

        label_id = int(result["prediction"][0])
        proba = float(result["fraud_probability"][0])

        mapping = {
            0: "Legitimate",
            1: "Fraud",
        }

        return {
            "label": mapping.get(label_id, str(label_id)),
            "proba": proba,
        }, 200

    except Exception as e:
        return f"Error calling SageMaker endpoint: {str(e)}", 500


# ─────────────────────────────────────────────────────────────────────
# SHAP explanation
# ─────────────────────────────────────────────────────────────────────
def display_explanation(input_df):
    """
    Render a SHAP waterfall plot for the fraud prediction.
    """

    try:
        explainer = load_shap_explainer(
            aws_bucket,
            posixpath.join("explainer", MODEL_INFO["explainer"]),
            os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"]),
        )

        shap_values = explainer(input_df)

        st.subheader("🔍 Decision Transparency SHAP")

        plt.figure(figsize=(10, 4))

        sv = shap_values[0]

        # Binary classifier SHAP values can return class-specific arrays.
        if hasattr(sv, "values") and getattr(sv.values, "ndim", 0) == 2:
            sv = sv[:, 1]

        shap.plots.waterfall(sv, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        if hasattr(sv, "values") and hasattr(sv, "feature_names"):
            top_feature = (
                pd.Series(sv.values, index=sv.feature_names)
                .abs()
                .idxmax()
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
st.title("💳 Fraud Detection — ML Deployment")

st.caption(
    "Tuned tree-based classifier trained on the IEEE-CIS transaction dataset. "
    "Predictions are served through an Amazon SageMaker endpoint."
)

with st.sidebar:
    st.header("AWS / Model Status")
    st.write(f"Endpoint: `{MODEL_INFO['endpoint']}`")
    st.write(f"Bucket: `{aws_bucket}`")
    st.write(f"Feature columns loaded: `{len(dataset.columns)}`")


with st.form("pred_form"):
    st.subheader("Transaction Inputs")

    cols = st.columns(2)
    user_inputs = {}

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
    # Start with the first row of X_train.csv so every model feature exists.
    # Then overwrite only the user-controlled fields from the UI.
    row = dataset.iloc[0:1].copy()

    for name, val in user_inputs.items():
        if name in row.columns:
            row[name] = val
        else:
            st.warning(
                f"Feature `{name}` was not found in X_train.csv. "
                "Check your MODEL_INFO inputs list."
            )

    res, status = call_model_api(row)

    if status == 200:
        c1, c2 = st.columns(2)

        c1.metric("Prediction", res["label"])
        c2.metric("Fraud Probability", f"{res['proba']:.2%}")

        if res["label"] == "Fraud":
            st.error(
                "⚠️ Flagged as fraudulent. Recommend manual review before authorization."
            )
        else:
            st.success("✅ Looks legitimate.")

        display_explanation(row)

    else:
        st.error(res)
