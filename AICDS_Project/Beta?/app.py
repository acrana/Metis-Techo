# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine
import pandas as pd
from predict_model import generate_alerts
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from data_preprocessing import connect_db, extract_data, prepare_model_data

app = Flask(__name__)
app.secret_key = 'your_strong_secret_key_here'  # Replace with a secure key in production

# Database Connection
engine = create_engine('sqlite:///clinical_data.db')

@app.route('/')
def index():
    """
    Home page displaying all patients.
    """
    # Display list of patients
    patients = pd.read_sql_table('Patients', engine)
    # Calculate age
    patients['age'] = (pd.to_datetime('today') - pd.to_datetime(patients['date_of_birth'])).dt.days // 365
    return render_template('index.html', patients=patients)

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    """
    Displays detailed information about a specific patient.
    """
    # Fetch patient details
    patient = pd.read_sql_query(f"SELECT * FROM Patients WHERE patient_id = {patient_id}", engine).iloc[0]
    
    # Fetch orders
    orders = pd.read_sql_query(f"""
        SELECT Orders.*, Medications.name as medication_name, Providers.first_name as provider_first, Providers.last_name as provider_last 
        FROM Orders 
        LEFT JOIN Medications ON Orders.medication_id = Medications.medication_id 
        LEFT JOIN Providers ON Orders.provider_id = Providers.provider_id 
        WHERE Orders.patient_id = {patient_id}
    """, engine)
    
    # Fetch lab results
    lab_results = pd.read_sql_query(f"SELECT * FROM Lab_Results WHERE patient_id = {patient_id} ORDER BY test_date DESC", engine)
    
    # Fetch alerts
    alerts = pd.read_sql_query(f"SELECT * FROM Alerts WHERE patient_id = {patient_id} AND resolved = 0", engine)
    
    return render_template('patient.html', patient=patient, orders=orders, lab_results=lab_results, alerts=alerts)

@app.route('/new_order/<int:patient_id>', methods=['GET', 'POST'])
def new_order(patient_id):
    """
    Allows users to create a new medication order for a patient.
    """
    if request.method == 'POST':
        # Extract form data
        medication_id = request.form.get('medication_id')
        dosage = request.form.get('dosage')
        frequency = request.form.get('frequency')
        duration = request.form.get('duration')
        provider_id = request.form.get('provider_id')
        order_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert new order into the database
        insert_query = f"""
            INSERT INTO Orders (patient_id, provider_id, medication_id, order_date, dosage, frequency, duration, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Active')
        """
        with engine.connect() as connection:
            connection.execute(insert_query, (patient_id, provider_id, medication_id, order_date, dosage, frequency, duration))
        
        # Fetch the newly inserted order_id
        order_id = pd.read_sql_query("SELECT last_insert_rowid() as order_id", engine).iloc[0]['order_id']
        
        # Prepare order data for alert generation
        new_order_data = {
            'patient_id': patient_id,
            'medication_id': int(medication_id),
            'dosage': dosage,
            'frequency': frequency,
            'duration': duration,
            'order_date': order_date
        }
        
        # Generate alerts based on the new order
        generate_alerts(new_order_data)
        
        flash('New order created successfully and alerts generated if any.', 'success')
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    # GET request: Render form to create a new order
    medications = pd.read_sql_table('Medications', engine)
    providers = pd.read_sql_table('Providers', engine)
    return render_template('new_order.html', patient_id=patient_id, medications=medications, providers=providers)

@app.route('/alerts')
def view_alerts():
    """
    Displays all active alerts across all patients.
    """
    # Display all active alerts
    alerts = pd.read_sql_query("""
        SELECT Alerts.*, Patients.first_name as patient_first, Patients.last_name as patient_last, Medications.name as medication_name 
        FROM Alerts 
        LEFT JOIN Patients ON Alerts.patient_id = Patients.patient_id 
        LEFT JOIN Medications ON Alerts.medication_id = Medications.medication_id 
        WHERE Alerts.resolved = 0
    """, engine)
    return render_template('alerts.html', alerts=alerts)

@app.route('/resolve_alert/<int:alert_id>', methods=['POST'])
def resolve_alert(alert_id):
    """
    Allows users to resolve an active alert.
    """
    # Mark the alert as resolved
    update_query = f"""
        UPDATE Alerts 
        SET resolved = 1 
        WHERE alert_id = {alert_id}
    """
    with engine.connect() as connection:
        connection.execute(update_query)
    
    # Optionally, add to Alert_History
    action_taken = request.form.get('action_taken')
    action_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    insert_history = f"""
        INSERT INTO Alert_History (alert_id, action_taken, action_date)
        VALUES (?, ?, ?)
    """
    with engine.connect() as connection:
        connection.execute(insert_history, (alert_id, action_taken, action_date))
    
    flash('Alert resolved successfully.', 'success')
    return redirect(url_for('view_alerts'))

def scheduled_monitoring():
    """
    This function runs periodically to check for any new alerts based on existing orders and lab results.
    """
    print("Running scheduled monitoring...")
    # Fetch all active orders
    orders = pd.read_sql_query("""
        SELECT Orders.*, Medications.name as medication_name 
        FROM Orders 
        LEFT JOIN Medications ON Orders.medication_id = Medications.medication_id 
        WHERE Orders.status = 'Active'
    """, engine)
    
    for _, order in orders.iterrows():
        order_data = {
            'patient_id': order.patient_id,
            'medication_id': order.medication_id,
            'dosage': order.dosage,
            'frequency': order.frequency,
            'duration': order.duration,
            'order_date': order.order_date
        }
        generate_alerts(order_data)
    print("Scheduled monitoring completed.")

if __name__ == '__main__':
    # Initialize and start the scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=scheduled_monitoring, trigger="interval", minutes=60)  # Runs every hour
    scheduler.start()
    
    try:
        app.run(debug=True)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
