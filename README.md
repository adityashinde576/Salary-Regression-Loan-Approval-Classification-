Salary (Regression) + Loan Approval (Classification) 
This project demonstrates a complete end-to-end Machine Learning workflow using:

Regression (Predict Salary)

Classification (Predict Loan Approval)

Preprocessing (Scaling + OneHotEncoding)

Multiple ML Models

Best Model Selection

Evaluation Metrics

Plots (Residual & Actual vs Predicted)

Confusion Matrix Heatmap

Project Overview

This project uses synthetic employee data with features like:

Income

Experience

Age

City

Salary (Generated)

Loan Approval (Yes/No)

We build:

 Regression Models (Predict Salary)

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Classification Models (Predict Loan Approval)

Logistic Regression

Decision Tree

Random Forest

KNN

Each model is wrapped inside a Pipeline (Preprocessing + Model).
Best model is selected automatically using metrics.
Key Features
1. Data Preprocessing

StandardScaler on numeric features

OneHotEncoder for City (categorical)

Label Encoding for LoanApproved

2. Regression Metrics

MAE

MSE

R² Score

 3. Classification Metrics

Accuracy

Confusion Matrix

Classification Report

4. Plots

Actual vs Predicted Salary

Residual Plot

Confusion Matrix Visualization

Project Structure
ML-Pipeline/
│
├── data_generation_and_preprocessing.py
├── regression_models.py
├── classification_models.py
├── visualization.py
├── README.md

🛠 Tech Stack
Category	Libraries
Data	NumPy, Pandas
ML Models	Scikit-learn
Preprocessing	StandardScaler, OneHotEncoder, LabelEncoder
Visualization	Matplotlib
How to Run the Project
1️Install Dependencies
pip install numpy pandas scikit-learn matplotlib
Run the Python File
python main.py
Output Includes

 Regression metrics
 Classification metrics
 Best model selection
 Visual graphs
 Confusion Matrix

Model Evaluation
Regression Models Compared
Model	MAE	MSE	R²

(This table auto-prints when you run code.)

Classification Models Compared
Model	Accuracy

(This table auto-prints when you run code.)

Best Model Selection

Best Regression Model (based on highest R²)
Best Classifier (based on highest Accuracy)

Project automatically prints:

Best model name

Best accuracy or R² score

Final predictions

Visualizations
1. Salary — Actual vs Predicted

A scatter plot showing model performance.

2. Salary Residual Plot

Shows errors to detect overfitting or trends.

3. Confusion Matrix for Classification

Shows TP, FP, FN, TN clearly.

Future Enhancements

Add hyperparameter tuning (GridSearchCV)
Add cross-validation
Deploy the model using Flask/FastAPI
Save model with joblib
