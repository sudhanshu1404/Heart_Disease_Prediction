def preprocess_input(form):
    return [
        float(form['age']),
        float(form['sex']),
        float(form['cp']),
        float(form['trestbps']),
        float(form['chol']),
        float(form['fbs']),
        float(form['restecg']),
        float(form['thalach']),
        float(form['exang']),
        float(form['oldpeak']),
        float(form['slope'])
    ]
