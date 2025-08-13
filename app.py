import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", message="is slow which will become errors in future versions.")

app = Flask(__name__)
CORS(app)

model = None
scaler = None
explainer = None
feature_names = None

def initialize_app():
    global model, scaler, explainer, feature_names
    try:
        model = tf.keras.models.load_model('breast_cancer_model.h5')
        scaler = joblib.load('scaler.pkl')
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return
    try:
        column_names = [
            'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
            'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
            'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        data = pd.read_csv('wdbc.data', header=None, names=column_names)
        data = data.drop('id', axis=1)
        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']
        feature_names = X.columns.tolist()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = scaler.transform(X_train)
        background_data = shap.sample(X_train_scaled, 100)
        explainer = shap.DeepExplainer(model, background_data)
        print("SHAP explainer created successfully.")
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or explainer is None:
        return jsonify({'error': 'Model, scaler, or explainer not initialized. Check server logs.'}), 500
    data = request.get_json(force=True)
    features = data.get('features')
    if features is None or len(features) != 30:
        return jsonify({'error': 'Invalid input. Please provide a list of 30 feature values.'}), 400

    try:
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
    
        prediction_proba = model.predict(scaled_data)[0][0]
        result = 'Malignant' if prediction_proba > 0.5 else 'Benign'
        shap_values = explainer.shap_values(scaled_data)[0]
        shap_values_list = shap_values.flatten().tolist()
        
        feature_importance = {name: val for name, val in zip(feature_names, shap_values_list)}

        response = {
            'prediction': result,
            'confidence_score': float(prediction_proba),
            'feature_importance': feature_importance
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500
if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000)
