# data_preprocessing.py
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def connect_db(db_path='clinical_data.db'):
    """
    Establishes a connection to the SQLite database.
    """
    engine = create_engine(f'sqlite:///{db_path}')
    return engine

def extract_data(engine):
    """
    Extracts data from all necessary tables in the database.
    """
    patients = pd.read_sql_table('Patients', engine)
    providers = pd.read_sql_table('Providers', engine)
    medications = pd.read_sql_table('Medications', engine)
    orders = pd.read_sql_table('Orders', engine)
    lab_results = pd.read_sql_table('Lab_Results', engine)
    med_lab_effects = pd.read_sql_table('Medication_Lab_Effects', engine)
    alerts = pd.read_sql_table('Alerts', engine)
    alert_history = pd.read_sql_table('Alert_History', engine)
    
    return {
        'patients': patients,
        'providers': providers,
        'medications': medications,
        'orders': orders,
        'lab_results': lab_results,
        'med_lab_effects': med_lab_effects,
        'alerts': alerts,
        'alert_history': alert_history
    }

def prepare_model_data(data):
    """
    Prepares data for model training by merging necessary tables and encoding categorical variables.
    
    Returns:
    - X: Feature matrix
    - y: Target vector
    - label_encoders: Dictionary of fitted LabelEncoders
    """
    # Merge Orders with Medications
    orders_med = pd.merge(
        data['orders'], 
        data['medications'], 
        on='medication_id', 
        how='left', 
        suffixes=('_order', '_med')
    )
    
    # Merge Orders with Patients
    orders_med_pat = pd.merge(
        orders_med, 
        data['patients'], 
        on='patient_id', 
        how='left', 
        suffixes=('', '_patient')
    )
    
    # Merge Orders with Providers to include 'role'
    orders_med_pat_prov = pd.merge(
        orders_med_pat, 
        data['providers'], 
        on='provider_id', 
        how='left', 
        suffixes=('', '_provider')
    )
    
    # Merge Orders with Lab_Results (most recent)
    lab_recent = data['lab_results'].sort_values(['patient_id', 'test_date']).groupby('patient_id').tail(1)
    orders_med_pat_prov_lab = pd.merge(
        orders_med_pat_prov, 
        lab_recent, 
        on='patient_id', 
        how='left', 
        suffixes=('', '_lab')
    )
    
    # Merge with Medication_Lab_Effects
    orders_med_pat_prov_lab_effects = pd.merge(
        orders_med_pat_prov_lab,
        data['med_lab_effects'],
        left_on=['medication_id', 'test_name'],
        right_on=['medication_id', 'lab_test_name'],
        how='left'
    )
    
    # Create Features for Modeling
    df = orders_med_pat_prov_lab_effects.copy()
    
    # Handle Missing Values
    df.fillna({'result_value': 0, 'impact': 'No Effect'}, inplace=True)
    
    # Encode Categorical Variables
    label_encoders = {}
    categorical_cols = ['gender', 'role', 'test_name', 'impact', 'status']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    
    # Feature Engineering: Time since last lab test
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['test_date'] = pd.to_datetime(df['test_date'])
    df['days_since_last_test'] = (df['order_date'] - df['test_date']).dt.days
    df['days_since_last_test'].fillna(df['days_since_last_test'].mean(), inplace=True)
    
    # Encode Additional Categorical Features: dosage, frequency, duration
    for col in ['dosage', 'frequency', 'duration']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    
    # Define Target Variable: Whether to trigger an alert (binary classification)
    df['trigger_alert'] = (
        (df['impact'] != 'No Effect') &
        (
            ((df['impact'] == 'Increase') & (df['result_value'] > df['threshold'])) |
            ((df['impact'] == 'Decrease') & (df['result_value'] < df['threshold']))
        )
    ).astype(int)
    
    # Select Features and Target
    feature_cols = [
        'dosage', 
        'frequency', 
        'duration', 
        'gender', 
        'role', 
        'test_name', 
        'impact', 
        'result_value', 
        'days_since_last_test'
    ]
    
    # Check for missing features
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f"Error: Missing features in DataFrame: {missing_features}")
        raise KeyError(f"Missing features: {missing_features}")
    
    X = df[feature_cols]
    y = df['trigger_alert']
    
    # Print class distribution
    print("Class distribution:")
    print(y.value_counts())
    
    return X, y, label_encoders

if __name__ == "__main__":
    engine = connect_db()
    data = extract_data(engine)
    try:
        X, y, label_encoders = prepare_model_data(data)
        print("Data preparation for model training completed successfully.")
        print("Feature Matrix (X):\n", X.head())
        print("\nTarget Vector (y):\n", y.head())
    except KeyError as e:
        print(f"Data preparation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

