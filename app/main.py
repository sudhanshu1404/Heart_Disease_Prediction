from flask import Flask, render_template, request
import numpy as np
import joblib
from utils.preprocess import preprocess_input

app = Flask(__name__)

# Load trained model
model = joblib.load('models/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        features = preprocess_input(data)
        prediction = model.predict([features])[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
