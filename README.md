# 🔥 Student Burnout Risk Prediction Using Machine Learning

## 📖 Overview

Student burnout has become a major concern in modern academic environments due to increasing academic pressure, poor sleep habits, financial stress, and psychological challenges.

This project uses Machine Learning to predict burnout risk levels among students by analyzing lifestyle, academic, and mental health related factors. The system classifies students into:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

The project also integrates Explainable AI (SHAP) to provide transparent predictions and personalized recommendations through an interactive Streamlit application.

---

## 🎯 Problem Statement

Traditional burnout assessment methods are often survey-based, subjective, and reactive. There is a need for a data-driven system that can identify students at risk before burnout becomes severe.

This project aims to provide an early-warning system using machine learning techniques.

---

## ✨ Key Features

* 🔥 Burnout Risk Prediction
* 📊 Multi-Class Classification (Low, Medium, High)
* 🤖 CatBoost Machine Learning Model
* 🧠 SHAP Explainability
* 🌐 Streamlit Web Application
* 📈 Data Visualization and EDA
* 💡 Personalized Wellness Recommendations

---

## 📂 Dataset Information

The project integrates three publicly available datasets:

### 📚 1. Student Lifestyle Dataset

Features:

* Study Hours
* Sleep Hours
* Social Hours
* Physical Activity
* GPA
* Academic Pressure

### 😔 2. Student Depression Dataset

Features:

* Gender
* Age
* Financial Stress
* Study Satisfaction

### ⚠️ 3. Stress Indicators Dataset

Features:

* Peer Competition
* Relationship Stress
* Sleep Problems
* Irritability

### 📌 Final Dataset Size

* 1,997 Records
* 14 Features

---

## 🧾 Features Used

| Category         | Features                                                                              |
| ---------------- | ------------------------------------------------------------------------------------- |
| 📚 Academic      | Study Hours, GPA, Academic Pressure, Study Satisfaction                               |
| 🏃 Lifestyle     | Sleep Hours, Social Hours, Physical Activity                                          |
| 👤 Demographic   | Age, Gender                                                                           |
| 🧠 Psychological | Financial Stress, Peer Competition, Relationship Stress, Sleep Problems, Irritability |

---

## 🎯 Target Variable Engineering

Burnout Risk is generated using domain-driven scoring rules:

* Study Hours > 8 → +1
* Sleep Hours < 6 → +1
* Academic Pressure > 3 → +1

### 📊 Risk Classification

| Score | Burnout Risk |
| ----- | ------------ |
| 0-1   | 🟢 Low       |
| 2     | 🟡 Medium    |
| 3     | 🔴 High      |

---

## ⚙️ Machine Learning Pipeline

1. 📥 Data Collection
2. 🧹 Data Cleaning
3. 🔗 Data Integration
4. 🏗️ Feature Engineering
5. 🎯 Burnout Label Generation
6. 🤖 Model Training
7. 📈 Model Evaluation
8. 🧠 SHAP Explainability
9. 🚀 Streamlit Deployment

---

## 🤖 Model Used

### CatBoost Classifier

#### Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Iterations    | 200   |
| Depth         | 6     |
| Learning Rate | 0.1   |

### Why CatBoost?

* ⚡ Handles structured tabular data efficiently
* 🔍 Captures non-linear relationships
* 🛠️ Requires minimal preprocessing
* 📈 Strong performance on mixed feature types

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 93%   |
| Precision | 93%   |
| Recall    | 93%   |
| F1 Score  | 92%   |

### 📈 Class-wise Performance

| Class  | Precision | Recall | F1   |
| ------ | --------- | ------ | ---- |
| Low    | 0.92      | 0.94   | 0.93 |
| Medium | 0.93      | 0.91   | 0.92 |
| High   | 0.94      | 0.89   | 0.91 |

---

## 🧠 Explainable AI (SHAP)

SHAP (SHapley Additive Explanations) is used to interpret model predictions.

It helps identify:

* 🔍 Which features contributed most
* ➕ Positive feature impacts
* ➖ Negative feature impacts
* 🎯 Why a student was classified into a particular risk category

This improves transparency and trust in the model.

---

## 📁 Project Structure

```text
Student-Burnout-Prediction-System
│
├── app.py
├── dataset.py
├── final.py
├── train_model.py
├── eda.py
│
├── data
│   ├── raw
│   └── processed
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

```bash
git clone https://github.com/Omm13/Student-Burnout-Prediction-System.git

cd Student-Burnout-Prediction-System

pip install -r requirements.txt

streamlit run app.py
```

---

## 🔮 Future Improvements

* 📡 Real-time student monitoring
* 📱 Mobile application integration
* ⏳ Time-series burnout prediction
* ⚖️ Improved handling of class imbalance
* 🌍 Multilingual support
* 🔒 Federated learning for privacy preservation

---

## 🛠️ Tech Stack

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 🤖 Scikit-Learn
* ⚡ CatBoost
* 🧠 SHAP
* 🌐 Streamlit
* 📊 Matplotlib

---

## 👥 Contributors

* Omm Miriyala
* Anagha Kharat
* Arpita Naik
* Vikas Pandey

---

## 🎓 Academic Project

This project was developed as part of the Mini Project curriculum for the Bachelor of Engineering in Computer Science & Engineering (Data Science).

**🎓Department:**  Computer Science & Engineering (Data Science)  
**🏫Institution:**  Vidyavardhini's College of Engineering and Technology  
**📚University:**  University of Mumbai
