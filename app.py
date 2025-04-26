from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import joblib
import pickle
import os
from tensorflow.keras.models import load_model
import shap
from scipy import sparse

app = Flask(__name__)

# Load saved assets
preprocessor = joblib.load('preprocessor.pkl')
best_model = load_model('best_model.h5')

# Load and prepare the background sample
background = np.load('background.npy', allow_pickle=True)
if sparse.issparse(background) or hasattr(background, "toarray"):
    background = background.toarray()
if background.ndim == 0:
    background = background.item()
elif background.ndim == 1:
    background = np.atleast_2d(background)

with open('shap_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

# Helper function to recursively convert numpy arrays to lists.
def convert_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(item) for item in obj]
    else:
        return obj

def predict_and_explain(input_data):
    # Convert the input dictionary to a DataFrame
    df = pd.DataFrame([input_data])
    # Preprocess the data using the saved preprocessor
    X_processed = preprocessor.transform(df)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    # Make prediction using the best model
    prediction = best_model.predict(X_processed)
    pred_label = "Readmitted" if prediction[0][0] > 0.5 else "Not Readmitted"
    
    # Generate SHAP explanation for the input sample.
    sample = X_processed  # sample shape (1, n_features)
    # Limit the number of samples to avoid index errors.
    shap_value_sample = explainer.shap_values(sample, nsamples=100)
    if isinstance(shap_value_sample, list):
        shap_value_sample = shap_value_sample[0]
    shap_value_sample = np.array(shap_value_sample)
    
    # Process expected value: if multi-output, take the first element.
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.item()
    
    # Generate the force plot.
    force_plot = shap.force_plot(
        expected_value,
        np.array(shap_value_sample),
        sample,
        feature_names=preprocessor.get_feature_names_out().tolist(),
        matplotlib=False
    )
    # Convert any numpy arrays in force_plot.data to lists.
    force_plot.data = convert_arrays_to_lists(force_plot.data)
    
    # Save the force plot as an HTML file using SHAP's built-in function.
    temp_file = "temp_shap.html"
    shap.save_html(temp_file, force_plot)
    with open(temp_file, 'r', encoding='utf-8') as f:
        shap_html = f.read()
    os.remove(temp_file)
    
    return pred_label, shap_html

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data. The keys must match the training data columns.
    input_data = {
        'age': request.form['age'],  # e.g., "[40-50)"
        'time_in_hospital': int(request.form['time_in_hospital']),
        'n_procedures': int(request.form['n_procedures']),
        'n_lab_procedures': int(request.form['n_lab_procedures']),
        'n_medications': int(request.form['n_medications']),
        'n_outpatient': int(request.form['n_outpatient']),
        'n_inpatient': int(request.form['n_inpatient']),
        'n_emergency': int(request.form['n_emergency']),
        'medical_specialty': request.form['medical_specialty'],
        'diag_1': request.form['diag_1'],
        'diag_2': request.form['diag_2'],
        'diag_3': request.form['diag_3'],
        'glucose_test': request.form['glucose_test'],
        'A1Ctest': request.form['A1Ctest'],
        'change': request.form['change'],
        'diabetes_med': request.form['diabetes_med']
    }
    prediction, shap_html = predict_and_explain(input_data)
    # Use Markup to safely render the HTML content.
    return render_template('result.html', prediction=prediction, shap_html=Markup(shap_html))

if __name__ == '__main__':
    app.run(debug=True)
