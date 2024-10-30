# predict_model.py
import tensorflow as tf
import joblib
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

def load_model_and_encoders(model_path='cdss_model.h5', encoders_path='label_encoders.pkl'):
    """
    Loads the trained TensorFlow model and label encoders.
    """
    model = tf.keras.models.load_model(model_path)
    label_encoders = joblib.load(encoders_path)
    return model, label_encoders

def preprocess_new_order(order_data, model, label_encoders, engine):
    """
    Preprocesses the new order data to prepare it for prediction.

    Parameters:
    - order_data (dict): Contains details of the new order.
    - model (tf.keras.Model): The trained TensorFlow model.
    - label_encoders (dict): Dictionary of label encoders for categorical features.
    - engine (sqlalchemy.Engine): Database engine.

    Returns:
    - alerts (list): List of generated alerts based on the prediction.
    """
    # Extract patient and medication information
    patient_id = order_data['patient_id']
    medication_id = order_data['medication_id']
    dosage = order_data['dosage']
    frequency = order_data['frequency']
    duration = order_data['duration']
    order_date = pd.to_datetime(order_data['order_date'])
    
    # Fetch patient's latest lab result relevant to the medication
    query = f"""
        SELECT lab_test_name, impact, threshold 
        FROM Medication_Lab_Effects 
        WHERE medication_id = {medication_id}
    """
    med_lab_effects = pd.read_sql_query(query, engine)
    
    alerts = []
    
    for _, effect in med_lab_effects.iterrows():
        test_name = effect['lab_test_name']
        impact = effect['impact']
        threshold = effect['threshold']
        alert_message = effect['alert_message']
        
        # Get the latest lab result for the test
        query_lab = f"""
            SELECT result_value 
            FROM Lab_Results 
            WHERE patient_id = {patient_id} AND test_name = '{test_name}'
            ORDER BY test_date DESC 
            LIMIT 1
        """
        lab_result = pd.read_sql_query(query_lab, engine)
        if not lab_result.empty:
            result_value = lab_result.iloc[0]['result_value']
            # Determine if alert should be triggered
            trigger = False
            if impact == 'Increase' and result_value > threshold:
                trigger = True
            elif impact == 'Decrease' and result_value < threshold:
                trigger = True
            
            if trigger:
                alerts.append({
                    'patient_id': patient_id,
                    'medication_id': medication_id,
                    'lab_test_name': test_name,
                    'alert_message': alert_message,
                    'alert_date': datetime.now(),
                    'resolved': False
                })
    
    return alerts

def generate_alerts(order_data):
    """
    Generates alerts based on the new order data.

    Parameters:
    - order_data (dict): Contains details of the new order.
    """
    model, label_encoders = load_model_and_encoders()
    engine = create_engine('sqlite:///clinical_data.db')
    
    # Preprocess the new order and get potential alerts
    alerts = preprocess_new_order(order_data, model, label_encoders, engine)
    
    # Insert alerts into the database
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        alerts_df.to_sql('Alerts', engine, if_exists='append', index=False)
        print(f"Generated {len(alerts)} alert(s) for patient ID {order_data['patient_id']}.")
    else:
        print("No alerts generated.")

if __name__ == "__main__":
    # Example new order data for testing
    new_order = {
        'patient_id': 1,
        'medication_id': 1,  # Amiodarone
        'dosage': '200 mg',
        'frequency': 'Once daily',
        'duration': '30 days',
        'order_date': '2024-04-15 14:00:00'
    }
    generate_alerts(new_order)
