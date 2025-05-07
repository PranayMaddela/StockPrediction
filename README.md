# 📈 ML/DL Stock Forecasting Engine

This project implements a Python-based forecasting engine for stock market trends using traditional Machine Learning and Deep Learning (ANN & LSTM) models. It supports both GUI- and CLI-based execution, enabling comparative analysis of raw continuous prices and binary movement trends.

## 🚀 Key Features

- 🔁 **Modular ML/DL automation pipeline**
- 🖥️ **GUI and CLI modes** for flexible execution
- 📊 **Metrics tracking**: Accuracy, F1-Score, ROC AUC
- 🧪 **Model comparison**: SVM, KNN, XGBoost, ANN, LSTM & more
- 📋 **HTML-based dashboards** for observability
- ☁️ **DevOps-ready**: scalable design for CI/CD and cloud deployment

## 🛠️ Tech Stack

- Python, Scikit-learn, TensorFlow/Keras  
- Tkinter (GUI), Matplotlib, Pandas  
- Automation logic for preprocessing and model execution

---

💡 **Project Goal:** Enable fast, repeatable stock trend predictions with a focus on performance tracking, modularity, and deployment readiness.


## 📁 Files Overview

| File | Purpose |
|------|---------|
| `StockPrediction.py` | Main GUI application (Tkinter) to run full workflow |
| `test.py` | Script to run predictions and evaluate models in console |
| `run.bat` | Launch script to open GUI directly |
| `SCREENS.docx` | Screenshots & project explanation |
| `*.csv` | Sample datasets (e.g., `HINDPETRO.NS.csv`, `BDD.csv`) |
| `binary_output.html` | Model results for binary-transformed data |
| `continuous_output.html` | Model results for raw continuous data |

## 🧠 Models Included

**Traditional ML:**
- SVM
- KNN
- Decision Tree
- Random Forest
- Logistic Regression
- AdaBoost
- XGBoost
- Naive Bayes

**Deep Learning:**
- ANN (Artificial Neural Network)
- LSTM (Long Short-Term Memory)

## 🚀 Features

- GUI-based dataset upload and execution
- Preprocessing and normalization
- Training on continuous and binary formats
- Evaluation metrics: Accuracy, F1-Score, ROC AUC
- LSTM actual vs predicted price visualization
- HTML comparison table generation

## 📦 How to Run

1. Install dependencies:

   pip install -r requirements.txt

2. Run the GUI:
   
   python StockPrediction.py

3. Or execute headless tests:
   
   python test.py

Alternatively, double-click run.bat to launch the GUI.

## 📁 Files Included
StockPrediction.py: Main GUI and model logic

test.py: Headless testing script

run.bat: Launch file

SCREENS.docx: Screenshots and walkthrough

*.csv: Sample datasets

binary_output.html, continuous_output.html: Model performance reports

## 📌 Dataset Format
CSV file must contain:

- Date: Date column

- Close: Stock closing price

Example:

Date,Close
2023-01-01,234.12
2023-01-02,235.80
...

## 🛠️ DevOps & Cloud Relevance

- Modular Python automation pipeline
- GUI/CLI dual-mode execution
- Metrics-based observability (Accuracy, F1-Score, ROC AUC)
- Outputs HTML dashboards for monitoring & validation
- Ready for containerization and cloud deployment
- Supports extension to CI/CD workflows

## 🧪 Output
Best Performing Model:
- Binary LSTM: 100% Accuracy, F1-Score, ROC AUC
- Continuous LSTM: ~96.7% Accuracy, strong F1 & ROC

Real-time LSTM vs Actual price plots

Comparative bar charts across all models

HTML summary of all accuracy and metrics
