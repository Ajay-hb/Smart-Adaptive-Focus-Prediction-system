# 🎯 Smart Adaptive Focus Prediction System (XGBoost + Streamlit)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🚀 Project Overview

The **Smart Adaptive Focus Prediction System** is an end-to-end machine learning application that predicts a user's focus score based on daily lifestyle factors such as sleep, stress, screen time, and exercise.

It also provides **personalized improvement suggestions** using model-driven insights.

---

## 🎯 Objective

- Predict user focus level (0–100)
- Provide actionable suggestions to improve focus
- Deliver real-time predictions via an interactive web app

---

## 🧠 Key Features

✔️ Synthetic dataset generation  
✔️ Feature engineering with interaction features  
✔️ XGBoost regression model  
✔️ Real-time prediction using Streamlit  
✔️ Personalized improvement suggestions  
✔️ Model visualization (feature importance, residuals)


---

## 🔍 Features Used

- Sleep Hours  
- Screen Time  
- Noise Level  
- Time of Day  
- Caffeine Intake  
- Stress Level  
- Exercise  
- Mood  
- Task Difficulty  

---

## 📊 Feature Engineering

- Sleep Category (Low / Medium / High)  
- Sleep × Stress Interaction  
- Caffeine × Sleep Interaction  
- Categorical Encoding using LabelEncoder  

---

## 🤖 Model

- Algorithm: **XGBoost Regressor**
- Optimized hyperparameters:
  - n_estimators = 300  
  - learning_rate = 0.05  
  - max_depth = 4  
- Handles non-linear relationships effectively

---



## 💻 Streamlit App

Interactive web app for real-time predictions.
<img width="1920" height="1529" alt="screencapture-localhost-8501-2026-04-24-13_33_59" src="https://github.com/user-attachments/assets/a8709f1b-294a-4962-a97a-44e8b77869d0" />


### 🔧 Features:
- User-friendly input interface  
- Instant focus score prediction  
- Personalized improvement suggestions  
- Visual insights dashboard  

---

## ▶️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/focus-prediction-system.git
cd focus-prediction-system
