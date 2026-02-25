from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
with open('models/kmeans_scaler.pkl', 'rb') as f:
    kmeans_scaler = pickle.load(f)

with open('models/linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)
with open('models/linear_scaler.pkl', 'rb') as f:
    linear_scaler = pickle.load(f)

# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------------------------------------------------
@app.route('/cluster', methods=['GET'])
def cluster_form():
    return render_template('cluster.html')

@app.route('/cluster_predict', methods=['POST'])
def cluster_predict():
    try:
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        fat = float(request.form['fat'])
        resting_bpm = float(request.form['resting_bpm'])
        avg_bpm = float(request.form['avg_bpm'])
        max_bpm = float(request.form['max_bpm'])
        session_dur = float(request.form['session_dur'])        
        frequency = float(request.form['frequency'])
        experience = float(request.form['experience'])

        # Build DataFrame with correct order (must match training)
        input_df = pd.DataFrame([[age, weight, height, bmi, fat,
                                   resting_bpm, avg_bpm, max_bpm,
                                   session_dur, frequency, experience]],
                                 columns=['Age', 'Weight (kg)', 'Height (m)',
                                          'BMI', 'Fat_Percentage',
                                          'Resting_BPM', 'Avg_BPM', 'Max_BPM',
                                          'Session_Duration (hours)',
                                          'Workout_Frequency (days/week)',
                                          'Experience_Level'])

        input_scaled = kmeans_scaler.transform(input_df.values)
        cluster = kmeans_model.predict(input_scaled)[0]

        risk_map = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
        result = f"Predicted Health Risk Cluster: {cluster} – {risk_map[cluster]}"
        return render_template('cluster.html', result=result)

    except Exception as e:
        return render_template('cluster.html', result=f"Error: {str(e)}")

# ----------------------------------------------------------------------
@app.route('/regression', methods=['GET'])
def regression_form():
    return render_template('regression.html')

@app.route('/regression_predict', methods=['POST'])
def regression_predict():
    try:
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        fat = float(request.form['fat'])
        resting_bpm = float(request.form['resting_bpm'])
        avg_bpm = float(request.form['avg_bpm'])
        max_bpm = float(request.form['max_bpm'])
        session_dur = float(request.form['session_dur'])
        frequency = float(request.form['frequency'])
        experience = float(request.form['experience'])
        water = float(request.form['water'])
        meals = float(request.form['meals'])
        carbs = float(request.form['carbs'])
        proteins = float(request.form['proteins'])
        fats = float(request.form['fats'])
        calories_diet = float(request.form['calories_diet'])
        cardio = float(request.form['cardio'])
        hiit = float(request.form['hiit'])
        strength = float(request.form['strength'])
        yoga = float(request.form['yoga'])

        hrr = max_bpm - resting_bpm
        pct_maxhr = avg_bpm / max_bpm
        lean_mass = weight * (1 - fat / 100)
        age2 = age ** 2                      
        bmi_x_freq = bmi * frequency

        num_benefits = 0
        num_muscle_groups = 0
        num_exercises = 0

        input_data = [[
            age, gender, weight, height, bmi, fat,
            resting_bpm, avg_bpm, max_bpm, session_dur,
            frequency, experience, water, meals,
            carbs, proteins, fats, calories_diet,
            hrr, pct_maxhr, lean_mass, age2, bmi_x_freq,
            num_benefits, num_muscle_groups, num_exercises,
            cardio, hiit, strength, yoga
        ]]

        columns = [
            'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage',
            'Resting_BPM', 'Avg_BPM', 'Max_BPM', 'Session_Duration (hours)',
            'Workout_Frequency (days/week)', 'Experience_Level', 'Water_Intake (liters)',
            'Daily meals frequency', 'Carbs', 'Proteins', 'Fats', 'Calories',
            'HRR', 'pct_maxHR', 'lean_mass_kg', 'Age2', 'BMI_x_Freq',
            'num_benefits', 'num_muscle_groups', 'num_exercises',
            'Workout_Type_Cardio', 'Workout_Type_HIIT', 'Workout_Type_Strength', 'Workout_Type_Yoga'
        ]

        input_df = pd.DataFrame(input_data, columns=columns)

        input_scaled = linear_scaler.transform(input_df.values)
        prediction = linear_model.predict(input_scaled)[0]

        result = f"Predicted Calories Burned: {prediction:.1f} kcal"
        return render_template('regression.html', result=result)

    except Exception as e:
        return render_template('regression.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)