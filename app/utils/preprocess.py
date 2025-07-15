# utils/preprocess.py

import numpy as np

def preprocess_input(data):
    """
    Convert form data into the correct numerical format for model input.
    Assumes all values are in string format from the form.
    """
    try:
        age = int(data['age'])
        sex = int(data['sex'])  # 1 = male, 0 = female
        cp = int(data['cp'])    # chest pain type (0–3)
        trestbps = int(data['trestbps'])
        chol = int(data['chol'])
        fbs = int(data['fbs'])  # fasting blood sugar > 120 mg/dl
        restecg = int(data['restecg'])  # resting electrocardiographic results
        thalach = int(data['thalach'])  # max heart rate achieved
        exang = int(data['exang'])      # exercise-induced angina
        oldpeak = float(data['oldpeak'])  # ST depression
        slope = int(data['slope'])        # slope of peak exercise ST segment
        ca = int(data['ca'])              # number of major vessels (0–3)
        thal = int(data['thal'])          # 1 = normal, 2 = fixed defect, 3 = reversible defect

        return np.array([
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ])
    except Exception as e:
        print(f"[Preprocessing Error] {e}")
        return np.zeros(13)
