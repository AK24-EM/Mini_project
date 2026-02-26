# Workout Analytics & Health Risk Predictor

An AI-powered Flask application designed to analyze physical activity data, predict calorie expenditure, and stratify health risks using Machine Learning.

## 🚀 Overview
This project leverages data-driven insights to help users understand their fitness performance and health status. By integrating regression and clustering models, it provides real-time predictions for calories burned and categorizes health risks based on physiological and lifestyle metrics.

---

## 💼 Business Problem Statement
In the modern era of sedentary lifestyles, lifestyle-related diseases (diabetes, hypertension, obesity) are rising. While wearable technology exists, there is often a **gap between data collection and actionable insight**. 

**The Problem:**
1. **Accessibility**: High-end metabolic testing is expensive.
2. **Personalization**: Generic "one-size-fits-all" fitness advice fails to account for individual physiological differences (BMI, Heart Rate Reserve, Experience Level).
3. **Preventative Insight**: Users often lack an early-warning system for health risks based on their current workout and dietary habits.

**The Solution:** This tool provides an accessible, personalized platform to quantify workout efficiency and assess health risks, empowering users to make informed lifestyle adjustments before chronic conditions develop.

---

## 📈 Economic Concepts Applied

### 1. Human Capital Theory
Investment in health is a primary form of **Human Capital**. By accurately tracking calories and health risks, individuals can optimize their "maintenance" (exercise and diet), leading to increased productivity, longevity, and higher lifetime earnings.

### 2. Information Asymmetry
In the fitness industry, there is a significant information gap between what a user *thinks* they are achieving and their actual metabolic output. This project reduces information asymmetry by providing empirical, model-backed feedback on calorie burn and health risk clusters.

### 3. Risk Stratification & Healthcare Cost Mitigation
By categorizing users into 'Low', 'Moderate', or 'High' risk clusters (K-Means), the project applies the economic principle of **Risk Stratification**. Identifying high-risk individuals early allows for targeted preventative care, potentially saving thousands in future healthcare costs (Negative Externality mitigation).

---

## 🤖 AI Techniques Used

### 1. Linear Regression (Calories Burned Prediction)
*   **Purpose**: Predict the exact amount of calories burned during a session.
*   **Key Features**: Age, Gender, Weight, Heart Rate Reserve (HRR), Lean Mass, Workout Type (HIIT, Strength, Yoga), and Calories from Diet.
*   **Preprocessing**: Features are standardized using `StandardScaler` to ensure uniform weighting.

### 2. K-Means Clustering (Health Risk Stratification)
*   **Purpose**: Group users into three distinct health risk profiles.
*   **Logic**: Uses unsupervised learning to identify patterns in BMI, Fat Percentage, Heart Rate, and Workout Frequency.
*   **Risk Map**: 
    *   **Cluster 0**: Low Risk (High frequency, healthy BMI)
    *   **Cluster 1**: Moderate Risk
    *   **Cluster 2**: High Risk (High fat %, low frequency)

### 3. Feature Engineering
Modern AI performance relies on high-quality features. We derived:
*   **BMI (Body Mass Index)**: Weight/Height².
*   **Heart Rate Reserve (HRR)**: Max BPM - Resting BPM (a key indicator of cardiovascular fitness).
*   **Lean Mass**: Calculated using Weight and Fat Percentage.
*   **Interaction Terms**: `BMI_x_Freq` (BMI multiplied by Workout Frequency).

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Mini_project
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python application.py
   ```
   Open `http://localhost:5001` in your browser.

---

## 🖥️ Usage
*   **Regression**: Enter your physiological data and workout details to get an estimate of calories burned.
*   **Clustering**: Enter health metrics to see which risk category you fall into and receive a risk level assessment.
