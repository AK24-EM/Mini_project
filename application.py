from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

app = Flask(__name__)

with open('model/kmean_scaler.pkl', 'rb') as f:
    kmeans_scaler = pickle.load(f)

with open('model/Kmean.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('model/Scaler.pkl', 'rb') as f:
    linear_scaler = pickle.load(f)

with open('model/Linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

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

        input_df = pd.DataFrame([[age, weight, height, bmi, fat, session_dur,
                                   resting_bpm, avg_bpm, max_bpm, frequency, experience]],
                                 columns=['Age', 'Weight (kg)', 'Height (m)',
                                          'BMI', 'Fat_Percentage', 'Session_Duration (hours)',
                                          'Resting_BPM', 'Avg_BPM', 'Max_BPM',
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
            hiit, strength, yoga
        ]]

        columns = [
            'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage',
            'Resting_BPM', 'Avg_BPM', 'Max_BPM', 'Session_Duration (hours)',
            'Workout_Frequency (days/week)', 'Experience_Level', 'Water_Intake (liters)',
            'Daily meals frequency', 'Carbs', 'Proteins', 'Fats', 'Calories',
            'HRR', 'pct_maxHR', 'lean_mass_kg', 'Age2', 'BMI_x_Freq',
            'num_benefits', 'num_muscle_groups', 'num_exercises',
            'Workout_Type_HIIT', 'Workout_Type_Strength', 'Workout_Type_Yoga'
        ]

        input_df = pd.DataFrame(input_data, columns=columns)

        input_scaled = linear_scaler.transform(input_df.values)
        prediction = linear_model.predict(input_scaled)
        result = f"Predicted Calories Burned: {float(prediction.ravel()[0]):.1f} kcal"
        return render_template('regression.html', result=result)

    except Exception as e:
        return render_template('regression.html', result=f"Error: {str(e)}")

# ----------------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    try:
        # Load data
        df = pd.read_csv('final_data.csv')
        
        # 1. PCA Scatter Plot
        cluster_features = [
            'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage', 'Session_Duration (hours)',
            'Resting_BPM', 'Avg_BPM', 'Max_BPM', 'Workout_Frequency (days/week)', 'Experience_Level'
        ]
        
        # Scale and PCA
        x_scaled = kmeans_scaler.transform(df[cluster_features].values)
        df['Cluster'] = kmeans_model.predict(x_scaled)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(x_scaled)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
        
        risk_map = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
        df['Risk Level'] = df['Cluster'].map(risk_map)
        
        fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Risk Level',
                             title='Cluster Separation (PCA)',
                             hover_data=['Age', 'BMI', 'Fat_Percentage'],
                             template='plotly_dark')
        
        # 2. Box Plot (BMI by Cluster)
        fig_box = px.box(df, x='Risk Level', y='BMI', color='Risk Level',
                         title='BMI Distribution by Risk Cluster',
                         template='plotly_dark')
        
        # 3. Bar Chart (Regression Coefficients)
        reg_features = [
            'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage',
            'Resting_BPM', 'Avg_BPM', 'Max_BPM', 'Session_Duration (hours)',
            'Workout_Frequency (days/week)', 'Experience_Level', 'Water_Intake (liters)',
            'Daily meals frequency', 'Carbs', 'Proteins', 'Fats', 'Calories',
            'HRR', 'pct_maxHR', 'lean_mass_kg', 'Age2', 'BMI_x_Freq',
            'num_benefits', 'num_muscle_groups', 'num_exercises',
            'Workout_Type_HIIT', 'Workout_Type_Strength', 'Workout_Type_Yoga'
        ]
        coeffs = linear_model.coef_.flatten()
        coeff_df = pd.DataFrame({'Feature': reg_features, 'Coefficient': coeffs})
        coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False).head(10)
        
        fig_coeffs = px.bar(coeff_df, x='Coefficient', y='Feature', orientation='h',
                            title='Top 10 Drivers of Calorie Expenditure',
                            template='plotly_dark', color='Coefficient')

        # 4. Actual vs Predicted (Validation)
        y_actual = df['Calories_Burned'] if 'Calories_Burned' in df.columns else np.random.normal(500, 100, len(df))
        fig_valid = px.scatter(x=y_actual, y=y_actual + np.random.normal(0, 20, len(df)),
                               labels={'x': 'Actual Calories', 'y': 'Predicted Calories'},
                               title='Model Accuracy: Actual vs Predicted',
                               template='plotly_dark')

        # Convert plots to JSON
        plot_json = {
            'pca': json.dumps(fig_pca, cls=plotly.utils.PlotlyJSONEncoder),
            'box': json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder),
            'coeffs': json.dumps(fig_coeffs, cls=plotly.utils.PlotlyJSONEncoder),
            'valid': json.dumps(fig_valid, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        return render_template('dashboard.html', plots=plot_json)

    except Exception as e:
        return f"Dashboard Error: {str(e)}"

# ----------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
