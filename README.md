# AI-CDSS

# AI Clinical Decision Support System (CDSS)

This project aims to develop an AI-driven Clinical Decision Support System (CDSS) to assist healthcare professionals in decision-making processes. This repository serves as a learning platform to explore and understand various tools and technologies such as Python, SQL, and TensorFlow.

## Project Purpose

The primary goal of this project is to learn and experiment with:

- **Python 3**
- **SQLite**: Database management
- **Pandas**: Data manipulation and analysis
- **TensorFlow/Keras**: Machine learning model building and training
- **scikit-learn**: Data preprocessing and utilities



By working on this project, I hope to gain hands-on experience with these technologies and eventually develop a functional CDSS.


- **Patient Data Management**: Stores and manages patient demographics, survey results, and adverse drug event (ADE) records.

- **Risk Prediction**: Uses a machine learning model to predict the risk of ADEs in patients.

- **Interactive Command-Line Interface**: Allows users to interact with the system, view patient information, and assess risk.

- **Data Preprocessing**: Implements data cleaning and preprocessing steps to prepare data for modeling.

- **Modular Code Structure**: Organized codebase for easy understanding and extensibility.



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

