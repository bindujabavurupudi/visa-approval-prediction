from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import pickle
import os
import math
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and columns
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'model_results.pkl')

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_columns = []

def convert_nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

@app.route("/")
def root():
    return jsonify({"status": "ok", "message": "Visa Approval Prediction API is running."})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        
        # Encoding maps
        continent_map = {"Asia": 0, "Europe": 1, "Africa": 2, "North America": 3, "South America": 4, "Oceania": 5}
        education_map = {"Bachelor's": 0, "Master's": 2, "Doctorate": 3, "High School": 1}
        yesno_map = {"Yes": 1, "No": 0}
        region_map = {"Midwest": 0, "Northeast": 1, "South": 3, "West": 2}
        unit_map = {"Hour": 0, "Week": 2, "Month": 3, "Year": 1}

        # Encode categorical
        encoded = {
            'continent': continent_map.get(data.get('continent', 'Asia'), 0),
            'education_of_employee': education_map.get(data.get('education_of_employee', "Bachelor's"), 0),
            'has_job_experience': yesno_map.get(data.get('has_job_experience', 'No'), 0),
            'requires_job_training': yesno_map.get(data.get('requires_job_training', 'No'), 0),
            'region_of_employment': region_map.get(data.get('region_of_employment', 'South'), 0),
            'unit_of_wage': unit_map.get(data.get('unit_of_wage', 'Year'), 0),
            'full_time_position': yesno_map.get(data.get('full_time_position', 'Yes'), 0)
        }

        # Standardize numerical
        no_of_employees = float(data.get('no_of_employees', 1))
        yr_of_estab = float(data.get('yr_of_estab', 2000))
        prevailing_wage = float(data.get('prevailing_wage', 50000))

        no_of_employees_log = np.log(max(no_of_employees, 1))
        yr_of_estab_log = np.log(max(yr_of_estab, 1900))
        prevailing_wage_sqrt = np.sqrt(max(prevailing_wage, 1))

        # StandardScaler values
        no_of_employees_log_stand = (no_of_employees_log - 5.5) / 1.2
        yr_of_estab_log_stand = (yr_of_estab_log - 7.5) / 0.2
        prevailing_wage_sqrt_stand = (prevailing_wage_sqrt - 500) / 100

        # Build final input
        model_input = {
            **encoded,
            'no_of_employees_log_stand': no_of_employees_log_stand,
            'yr_of_estab_log_stand': yr_of_estab_log_stand,
            'prevailing_wage_sqrt_stand': prevailing_wage_sqrt_stand
        }
        
        df = pd.DataFrame([model_input])
        df = df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(df)[0]
        label_map = {0: "Denied", 1: "Certified"}
        readable_pred = label_map.get(int(prediction), str(prediction))
        
        return jsonify({"prediction": readable_pred, "raw": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/model/results", methods=["GET"])
def model_results():
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        results = convert_nan_to_none(results)
        results["accuracy"] = float(results["accuracy"])
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model/info", methods=["GET"])
def model_info():
    try:
        info = {
            "model_file": os.path.basename(MODEL_PATH),
            "columns_file": os.path.basename(COLUMNS_PATH),
            "results_file": os.path.basename(RESULTS_PATH),
            "model_columns": model_columns
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
