📈 Stock Market Trend Prediction using ML & DL (Continuous vs Binary Analysis)
This project performs comparative analysis between traditional machine learning and deep learning models to predict stock price trends. It supports both continuous stock values and binary transformations (1 for increase, -1 for decrease), enabling a dual-mode evaluation.

🗂 Dataset
Format: CSV

Required Columns: Date, Close

Used Stocks: Example includes HINDPETRO.NS.csv, BDD.csv, etc.

🧠 Models Used
Traditional ML Models:

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Logistic Regression

AdaBoost

XGBoost

Naive Bayes

Deep Learning Models:

Artificial Neural Network (ANN)

Long Short-Term Memory (LSTM)

⚙️ Project Features
Upload and preprocess datasets via GUI (Tkinter)

Generate binary vs continuous datasets

Train and test multiple ML/DL models on each

Output performance in terms of Accuracy, F1-Score, and ROC AUC

Visual comparison graphs and HTML-based performance tables

📊 Performance Highlights
From binary_output.html and continuous_output.html:

Binary LSTM: 100% Accuracy, F1-Score, and ROC AUC

Continuous LSTM: ~96.7% Accuracy with high F1-Score

ANN and Decision Trees also showed strong performance

🖥 How to Run
Double-click run.bat to launch the GUI.

Use GUI buttons in sequence:

Upload Stock Dataset

Preprocess Dataset

Run Continuous Prediction

Run Binary Prediction

Comparison Graph

View Comparison Table

📁 Files Included
StockPrediction.py: Main GUI and model logic

test.py: Headless testing script

run.bat: Launch file

SCREENS.docx: Screenshots and walkthrough

*.csv: Sample datasets

binary_output.html, continuous_output.html: Model performance reports

🧪 Output
Real-time LSTM vs Actual price plots

Comparative bar charts across all models

HTML summary of all accuracy and metrics
