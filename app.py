from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier  # Add this import
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier  # Add this import
import os

app = Flask(__name__)

# Load models
with open('ml model/logreg_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)
with open('ml model/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('ml model/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('ml model/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('ml model/dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('ml model/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('ml model/xgb_model.json')
with open('ml model/adaboost_model.pkl', 'rb') as f:  # Add this line
    adaboost_model = pickle.load(f)
# Load or train ensemble models (example: train and save first, then load as below)
with open('ml model/brf_model.pkl', 'rb') as f:
    brf_model = pickle.load(f)
with open('ml model/easy_ensemble_model.pkl', 'rb') as f:
    easy_ensemble_model = pickle.load(f)

# Load your dataset
df = pd.read_csv('dataset/test-2.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Calculate accuracy for each model using the entire dataset
model_accuracies = {
    'logreg': accuracy_score(y, logreg_model.predict(X)),
    'svm': accuracy_score(y, svm_model.predict(X)),
    'knn': accuracy_score(y, knn_model.predict(X)),
    'rf': accuracy_score(y, rf_model.predict(X)),
    'dt': accuracy_score(y, dt_model.predict(X)),
    'gb': accuracy_score(y, gb_model.predict(X)),
    'xgb': accuracy_score(y, xgb_model.predict(X)),
    'adaboost': accuracy_score(y, adaboost_model.predict(X)),  # Add this line
    'brf': accuracy_score(y, brf_model.predict(X)),  # Balanced Random Forest
    'easy_ensemble': accuracy_score(y, easy_ensemble_model.predict(X)),  # Easy Ensemble
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/visualizations.html')
def visualizations():
    return render_template('visualizations.html')

@app.route('/analysis.html')
def analysis():
    return render_template('analysis.html')

@app.route('/amount-trends.html')
def amount_trends():
    return render_template('amount-trends.html')

@app.route('/feature.html')
def feature():
    return render_template('feature.html')

@app.route('/theory.html')
def theory():
    return render_template('theory.html')

@app.route('/model.html')
def model():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    model_name = data.get('model', 'logreg')  # default to logistic regression

    if model_name == 'svm':
        model = svm_model
    elif model_name == 'knn':
        model = knn_model
    elif model_name == 'rf':
        model = rf_model
    elif model_name == 'dt':
        model = dt_model
    elif model_name == 'gb':
        model = gb_model
    elif model_name == 'xgb':
        model = xgb_model
    elif model_name == 'adaboost':
        model = adaboost_model
    elif model_name == 'brf':
        model = brf_model
    elif model_name == 'easy_ensemble':
        model = easy_ensemble_model
    else:
        model = logreg_model

    pred = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0, 1]
    else:
        prob = float(model.decision_function(features)[0])
    acc = model_accuracies.get(model_name, model_accuracies['logreg'])
    return jsonify({'prediction': int(pred), 'probability': float(prob), 'accuracy': float(acc)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


