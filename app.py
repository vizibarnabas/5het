from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import logging
import sys
import mlflow
from mlflow.tracking import MlflowClient

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5102"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = Flask(__name__)

# --- Load Model from MLflow Model Registry at Startup ---
mlflow_model = None
MODEL_NAME = "CreditApprovalModel"
MODEL_STAGE = "staging" # Changed to lowercase

try:
    logging.info(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_STAGE}'...")
    # Use the newer '@' syntax for aliases
    mlflow_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_STAGE}")
    logging.info("MLflow model loaded successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to load model from MLflow Model Registry. The application cannot serve predictions. Error: {e}")

@app.route("/predict", methods=['POST'])
def predict():
    if not mlflow_model:
        logging.error("Prediction endpoint called, but MLflow model is not loaded.")
        return jsonify({"error": "Model is not available. Check server startup logs for critical errors."}), 503

    try:
        json_data = request.get_json()
        if 'data' not in json_data:
            return jsonify({"error": "Missing 'data' key in JSON payload"}), 400

        input_data = json_data.get('data')
        
        # Create DataFrame from input
        columns = [f'col{i}' for i in range(15)]
        df = pd.DataFrame(input_data, columns=columns)

        # --- Data Cleaning (must match training) ---
        df.replace('?', np.nan, inplace=True)
        numerical_cols = ["col1", "col2", "col7", "col10", "col13", "col14"]
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        logging.info(f"Received {len(input_data)} records for prediction.")

        # --- Predict using the loaded MLflow model ---
        # The MLflow model pipeline handles both preprocessing and prediction
        prediction = mlflow_model.predict(df)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route("/", methods=['GET'])
def health_check():
    return "Credit Card Approval Prediction API is running. Use the /predict endpoint for predictions."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
