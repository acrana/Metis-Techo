# AI-CDSS

# AI Clinical Decision Support System 

This project aims to develop an AI-driven Clinical Decision Support System (CDSS) to assist healthcare professionals in decision-making processes. This repository serves as a learning platform to explore and understand various tools and technologies such as Python, SQL, and TensorFlow.

## Project Purpose

The primary goal of this project is to learn and experiment with:

- **Python 3**
- **SQLite**: Database management
- **Pandas**: Data manipulation and analysis
- **TensorFlow/Keras**: Machine learning model building and training
- **scikit-learn**: Data preprocessing and utilities



  Übersehenjisatsu



![GetImage](https://github.com/user-attachments/assets/2cc4101e-54b5-4b97-99b7-212758295f8b)
```mermaid
graph TD
    UI[User Interface main.py] -->|Interacts with| DAL[Data Access Layer data_access.py]
    DAL -->|Queries| DB[SQLite Database clinical_decision_support.db]
    UI -->|Initiates| RA[Risk Assessment risk_assessment.py]
    RA -->|Uses| DP[Data Preprocessing preprocessing.py]
    DP -->|Processes Data for| ML[Machine Learning Model model.h5]
    ML -->|Provides Risk Prediction to| RA
    RA -->|Returns Results to| UI
    MT[Model Training model_training.py] -->|Saves Model to| ML



