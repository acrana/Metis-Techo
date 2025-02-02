# CLABSI Prediction Model Pipeline

## Introduction

Central Line-Associated Bloodstream Infection (CLABSI) is a serious, life-threatening complication in critically ill patients. This project leverages data from the MIMIC-IV database to build an explainable machine learning model for predicting CLABSI risk. By identifying key factors that contribute to infection risk, the model aims to support early intervention and improve patient outcomes.

## Data Preparation

- **Data Source:**  
  MIMIC-IV database

- **Data Processing:**  
  - Extracted cohorts of CLABSI cases and matched controls.
  - Cleaned the dataset by removing duplicate ICU stays and handling out-of-range values (e.g., negative dressing change times).
  - Focusing on central line characteristics such as `line_site`, `site_category`, and `risk_category`.

- **Feature Engineering:**  
  The final feature set includes patient demographics, lab values, central line characteristics, clinical care metrics, and severity scores.
  
## Model Training and Evaluation

- **Model Selection:**  
  Multiple classifiers were evaluated (e.g., Logistic Regression, Random Forest, XGBoost). XGBoost was selected based on its superior ROC AUC score.

- **Hyperparameter Tuning:**  
  GridSearchCV was used to fine-tune parameters such as the number of trees, max depth, and learning rate.  
  **Best Parameters:**  
  - `colsample_bytree`: 0.8  
  - `learning_rate`: 0.1  
  - `max_depth`: 7  
  - `n_estimators`: 200  
  - `subsample`: 1.0  

- **Performance:**  
  The final model achieved a test accuracy of approximately 87% and a ROC AUC of about 0.70.

- **Final Model:**  
  After validation, the model was retrained on the full dataset and saved as `final_xgb_model.pkl` for deployment.

## Deployment

The final model is saved and ready for integration. Future work includes wrapping this model into an API (using Flask or FastAPI) that can accept clinical data in FHIR format and return CLABSI risk predictions.


![shap_summary_beeswarm_custom](https://github.com/user-attachments/assets/b16a0bba-9ffc-4048-8866-9eced9e361f6)
