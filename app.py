from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# Load the best model and metadata
MODEL_PATH = "best_model.pkl"
METADATA_PATH = "model_metadata.json"
SCALER_PATH = "scaler.pkl"

# Global variables to store loaded models
best_model = None
scaler = None
model_metadata = None

def load_model_and_metadata():
    """Load the best model and its metadata"""
    global best_model, scaler, model_metadata
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            best_model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Model file {MODEL_PATH} not found. Please run train_model.py first.")
            return False
        
        # Load scaler if exists
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            print(f"Model metadata loaded: {model_metadata['model_name']} with accuracy {model_metadata['accuracy']:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    """Home page with model information"""
    return render_template('index.html', metadata=model_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if best_model is None:
            return jsonify({"error": "Model not loaded. Please run train_model.py first."}), 500
        
        # Get features from request
        data = request.get_json()
        features = data.get('features')
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Apply scaling if scaler exists
        if scaler is not None:
            features_array = scaler.transform(features_array)
        
        # Make prediction
        prediction = best_model.predict(features_array)
        prediction_proba = best_model.predict_proba(features_array)
        
        # Prepare response
        response = {
            "prediction": int(prediction[0]),
            "probability": {
                "class_0": float(prediction_proba[0][0]),
                "class_1": float(prediction_proba[0][1])
            },
            "model_info": {
                "model_name": model_metadata['model_name'] if model_metadata else "Unknown",
                "accuracy": model_metadata['accuracy'] if model_metadata else "Unknown"
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_metadata is None:
        return jsonify({"error": "Model metadata not available"}), 404
    
    return jsonify(model_metadata)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": best_model is not None,
        "scaler_loaded": scaler is not None,
        "metadata_loaded": model_metadata is not None
    }
    return jsonify(status)

if __name__ == "__main__":
    # Load model and metadata on startup
    if load_model_and_metadata():
        print("Flask app starting...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please run train_model.py first.")