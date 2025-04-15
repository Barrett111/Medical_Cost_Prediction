from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
import warnings
import time
import traceback
import os

# Initialize at module level (loads only once)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'Data' / 'pmjay_rates.parquet'  # Using Parquet for faster I/O
MODEL_DIR = BASE_DIR / 'ML_Models'

# Global variables for cached data
_DF = None
_MODELS = {}
_SCALERS = {}

def initialize_globals():
    """Load all data and models once at startup"""
    global _DF, _MODELS, _SCALERS
    
    if _DF is None:
        print("Loading data...", file=sys.stderr)
        _DF = pd.read_parquet(str(DATA_PATH))  # Read Parquet instead of CSV
        print("Columns in DataFrame:", _DF.columns.tolist(), file=sys.stderr)

        # Create index for faster lookup
        _DF.set_index('procedure_name', inplace=True)

        # Generate derived columns
        _DF['category_code'] = pd.factorize(_DF['category'])[0]
        _DF['procedure_code'] = pd.factorize(_DF.index)[0]
        _DF['severity'] = _DF['package_code'].str.extract(r'(\d+)')[0].fillna('1').astype(int)
        _DF['govt_tier_diff'] = _DF['tier1_cost'] - _DF['tier3_cost']
        _DF['private_tier_diff'] = _DF['private_tier1'] - _DF['private_tier3']

    if not _MODELS:
        print("Loading models...", file=sys.stderr)
        _MODELS = {
            'govt': joblib.load(str(MODEL_DIR / 'govt_cost_predictor.pkl')),
            'private': joblib.load(str(MODEL_DIR / 'private_cost_predictor.pkl'))
        }
        
        _SCALERS = {
            'govt': joblib.load(str(MODEL_DIR / 'govt_scaler.pkl')),
            'private': joblib.load(str(MODEL_DIR / 'private_scaler.pkl'))
        }

# Initialize on import
initialize_globals()

def predict_medical_cost(procedure_name, hospital_type='private', city_tier=1, metro=False):
    """Simplified and optimized prediction function"""
    try:
        # Check if procedure exists using index (faster)
        if procedure_name not in _DF.index:
            available = _DF.index.unique().tolist()
            return {
                'error': f"Procedure not found. Available: {available[:10]}... (total: {len(available)})",
                'available_procedures': available
            }
            
        procedure = _DF.loc[procedure_name]
        if isinstance(procedure, pd.DataFrame):
            procedure = procedure.iloc[0]  # Take the first match
        
        if hospital_type == 'govt':
            X = np.array([
                procedure['category_code'],
                procedure['procedure_code'],
                procedure['severity'],
                procedure['govt_tier_diff']
            ]).reshape(1, -1)
            
            X_scaled = _SCALERS['govt'].transform(X)
            pred = _MODELS['govt'].predict(X_scaled)[0]
            
        else:  # private
            X = np.array([
                procedure['category_code'],
                procedure['procedure_code'],
                procedure['severity'],
                procedure['private_tier_diff'],
                procedure['govt_tier_diff']
            ]).reshape(1, -1)
            
            X_scaled = _SCALERS['private'].transform(X)
            pred = _MODELS['private'].predict(X_scaled)[0]
        
        return {
            'predicted_cost': round(float(pred), 2),
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    procedure_name = data.get('procedure')
    hospital_type = data.get('hospitalType', 'private')
    city_tier = int(data.get('cityTier', 1))
    metro = data.get('metro', False)
    
    result = predict_medical_cost(procedure_name, hospital_type, city_tier, metro)
    return jsonify(result)

#if __name__ == '__main__':
   #app.run(debug=True)

   

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets this automatically
    app.run(debug=False, host='0.0.0.0', port=port)
