def call_model_api(input_df):
    """
    Send one row of feature data to the SageMaker endpoint.

    Uses split format:
    {
        "columns": [...],
        "data": [[...]]
    }

    This keeps the exact feature names and order attached to the request,
    which helps the SageMaker inference handler rebuild the model input
    correctly.
    """

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    # Clean values so JSON/SageMaker does not receive NaN, inf, or NumPy objects.
    clean_df = input_df.copy()
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.where(pd.notnull(clean_df), 0.0)

    # Try to convert all columns to numeric. Non-numeric values become 0.0.
    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.fillna(0.0)

    payload = {
        "columns": list(clean_df.columns),
        "data": clean_df.astype(float).values.tolist(),
    }

    try:
        result = predictor.predict(payload)

        label_id = int(result["prediction"][0])
        proba = float(result["fraud_probability"][0])

        # Normalize probability for Streamlit percent formatting.
        # Example: if endpoint returns 6.5107, treat it as 6.5107% -> 0.065107.
        if proba > 1:
            proba = proba / 100

        # Keep probability safely between 0 and 1.
        proba = max(0.0, min(proba, 1.0))

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
