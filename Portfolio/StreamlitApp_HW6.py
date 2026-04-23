import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load



# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource  # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features()

# HW6 (regression) feature set for target ticker NFLX:
#   - target ticker's sentiment + 3 lags
#   - other tickers' sentiment (everything in the allowed list except NFLX)
FEATURE_KEYS = [
    'sentiment_textblob', 'sentiment_lag1', 'sentiment_lag2', 'sentiment_lag3',
    'AAPL', 'MSFT', 'AMZN', 'GOOG', 'WMT', 'JPM', 'TSLA', 'ADBE'
]

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_sentiment.shap',
    "pipeline": 'finalized_sentiment_model.tar.gz',
    "keys": FEATURE_KEYS,
    # Sentiment scores are bounded in [-1, 1]; default to neutral (0.0)
    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in FEATURE_KEYS
    ]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)

# Prediction Logic
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        # Regression: endpoint returns a numeric next-day return
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # Format as a percentage return with 4 decimal places
        return f"{float(pred_val) * 100:.4f}% (raw: {float(pred_val):.6f})", 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Pipeline is [imputer, scaler, model] -> strip the last step (model) for SHAP input
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names = best_pipeline[:-1].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    # Regression waterfall (single-output model)
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)

    # Top feature by absolute SHAP value (regression)
    top_feature = pd.Series(
        shap_values[0].values,
        index=shap_values[0].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment — Next-Day Return Prediction")
st.caption("Target ticker: **NFLX** | Model: XGBoost Regressor on sentiment features")

with st.form("pred_form"):
    st.subheader("Inputs (sentiment scores, range -1 to 1)")
    cols = st.columns(3)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 3]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Predicted Next-Day Return", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
