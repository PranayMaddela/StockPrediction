# 📈 Stock Market Trend Prediction using ML & DL (Continuous vs Binary Analysis)

This project compares traditional machine learning and deep learning models to predict stock market trends using both continuous price values and binary transformations (based on up/down movement).

## 📊 Project Objective

Evaluate and compare 9 ML algorithms and 2 DL models (ANN & LSTM) across:
- **Continuous Price Data**
- **Binary Movement Data** (1 if current price > previous, else -1)

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


## 🧪 Output
Best Performing Model:
- Binary LSTM: 100% Accuracy, F1-Score, ROC AUC
- Continuous LSTM: ~96.7% Accuracy, strong F1 & ROC

Real-time LSTM vs Actual price plots

Comparative bar charts across all models

HTML summary of all accuracy and metrics
