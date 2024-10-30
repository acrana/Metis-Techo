# data_preprocessing.py
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def connect_db(db_path='clinical_data.db'):
    engine = create_engine(f'sqlite:///{db_path}')
    return engine

def extract_data(engine):
    # Extract Patients
    patients = pd.read_sql_table('Patients', engine)
    
    # Extract Providers
    providers = pd.read_sql_table('Providers', engine)
    
    # Extract Medications
    medications = pd.read_sql_table('Medications', engine)
    
    # Extract Orders
    orders = pd.read_sql_table('Orders', engine)
    
    # Extract Lab_Results
    lab_results = pd.read_sql_table('Lab_Results', engine)
    
    # Extract Medication_Lab_Effects
    med_lab_effects = pd.read_sql_table('Medication_Lab_Effects', engine)
    
    # Extract Alerts
    alerts = pd.read_sql_table('Alerts', engine)
    
    # Extract Alert_History
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

def preprocess_data(data):
    # Merge Orders with Medications
    orders_med = pd.merge(data['orders'], data['medications'], on='medication_id', how='left', suffixes=('_order', '_med'))
    
    # Merge Orders with Patients
    orders_med_pat = pd.merge(orders_med, data['patients'], on='patient_id', how='left', suffixes=('', '_patient'))
    
    # Merge Orders with Lab_Results
    lab_recent = data['lab_results'].sort_values(['patient_id', 'test_date']).groupby('patient_id').tail(1)
    orders_med_pat_lab = pd.merge(orders_med_pat, lab_recent, on='patient_id', how='left', suffixes=('', '_lab'))
    
    # Merge with Medication_Lab_Effects
    orders_med_pat_lab_effects = pd.merge(
        orders_med_pat_lab,
        data['med_lab_effects'],
        left_on=['medication_id', 'test_name'],
        right_on=['medication_id', 'lab_test_name'],
        how='left'
    )
    
    # Create Features for Modeling
    df = orders_med_pat_lab_effects.copy()
    
    # Handle Missing Values if any
    df.fillna({'result_value': 0, 'impact': 'No Effect'}, inplace=True)
    
    # Encode Categorical Variables
    label_encoders = {}
    categorical_cols = ['gender', 'role', 'test_name', 'impact', 'status']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Feature Engineering
    # Example: Time since last lab test
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['test_date'] = pd.to_datetime(df['test_date'])
    df['days_since_last_test'] = (df['order_date'] - df['test_date']).dt.days
    df['days_since_last_test'].fillna(df['days_since_last_test'].mean(), inplace=True)
    
    # Target Variable: Whether to trigger an alert (binary classification)
    # For simplicity, we'll assume that an alert should be triggered if:
    # - The current medication impacts a lab test
    # - The patient's latest lab result crosses the threshold
    df['trigger_alert'] = (
        (df['impact'] != 'No Effect') &
        (
            (df['impact'] == 'Increase') & (df['result_value'] > df['threshold']) |
            (df['impact'] == 'Decrease') & (df['result_value'] < df['threshold'])
        )
    ).astype(int)
    
    # Select Features and Target
    feature_cols = ['dosage', 'frequency', 'duration', 'gender', 'role', 'test_name', 'impact', 'result_value', 'days_since_last_test']
    X = df[feature_cols]
    y = df['trigger_alert']
    
    # Convert categorical text fields to numerical if not already
    # Assuming 'dosage', 'frequency', 'duration' are categorical or can be encoded
    # For simplicity, we'll use Label Encoding for these as well
    for col in ['dosage', 'frequency', 'duration']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoders

if __name__ == "__main__":
    engine = connect_db()
    data = extract_data(engine)
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(data)
    print("Data preprocessing completed successfully.")
    print("Training Features:\n", X_train.head())
    print("\nTraining Labels:\n", y_train.head())
