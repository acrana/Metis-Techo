import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from risk_predictor import RiskPredictor
from datetime import datetime, timedelta

def test_risk_predictor(db_path="C:/Users/acran/OneDrive/Desktop/Projects/GPT/AI CDSS/TESTING - Copy/ai_cdss.db"):
    """Comprehensive test suite for risk prediction"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    predictor = RiskPredictor()
    
    def print_risk_scores(desc, scores):
        print(f"\n{desc}:")
        print(f"ADE Risk: {scores['ade_risk']:.1%}")
        print(f"Interaction Risk: {scores['interaction_risk']:.1%}")
        print(f"Overall Risk: {scores['overall_risk']:.1%}")
    
    # Test Case 1: Patient with no medications
    print("\n=== Testing Patient with No Medications ===")
    cursor.execute("""
        SELECT patient_id FROM Patients 
        WHERE patient_id NOT IN (
            SELECT DISTINCT patient_id 
            FROM Patient_Medications 
            WHERE dosage != 'DISCONTINUED'
        ) LIMIT 1
    """)
    clean_patient = cursor.fetchone()['patient_id']
    scores = predictor.predict(cursor, clean_patient)
    print_risk_scores("Base scores (should be minimal)", scores)

    # Test Case 2: Single common medication
    print("\n=== Testing Single Common Medication ===")
    medications = [
        (3, "Metoprolol"),   # Common beta blocker
        (1, "Aspirin"),      # Common antiplatelet
        (14, "Lisinopril")   # Common ACE inhibitor
    ]
    
    for med_id, med_name in medications:
        scores = predictor.predict(cursor, clean_patient, med_id)
        print_risk_scores(f"Single medication ({med_name})", scores)

    # Test Case 3: Patient with medication history
    print("\n=== Testing Patient with ADE History ===")
    cursor.execute("""
        SELECT DISTINCT am.patient_id
        FROM ADE_Monitoring am
        LIMIT 1
    """)
    ade_patient = cursor.fetchone()['patient_id']
    
    # Get their ADE history
    cursor.execute("""
        SELECT m.name, am.description, am.timestamp
        FROM ADE_Monitoring am
        JOIN Medications m ON am.medication_id = m.medication_id
        WHERE am.patient_id = ?
        ORDER BY am.timestamp DESC
    """, (ade_patient,))
    
    print("\nADE History:")
    for ade in cursor.fetchall():
        print(f"• {ade['timestamp']}: {ade['name']} - {ade['description']}")
    
    scores = predictor.predict(cursor, ade_patient)
    print_risk_scores("Base scores with ADE history", scores)

    # Test Case 4: Complex interaction scenario
    print("\n=== Testing Complex Interactions ===")
    cursor.execute("""
        SELECT patient_id, COUNT(*) as med_count
        FROM Patient_Medications
        WHERE dosage != 'DISCONTINUED'
        GROUP BY patient_id
        HAVING med_count >= 3
        LIMIT 1
    """)
    complex_patient = cursor.fetchone()['patient_id']
    
    # Get their current medications
    cursor.execute("""
        SELECT m.medication_id, m.name
        FROM Patient_Medications pm
        JOIN Medications m ON pm.medication_id = m.medication_id
        WHERE pm.patient_id = ?
        AND pm.dosage != 'DISCONTINUED'
    """, (complex_patient,))
    
    print("\nCurrent Medications:")
    current_meds = cursor.fetchall()
    for med in current_meds:
        print(f"• {med['name']}")
    
    scores = predictor.predict(cursor, complex_patient)
    print_risk_scores("Base scores with multiple medications", scores)
    
    # Test adding high-risk medication
    high_risk_meds = [
        (9, "Digoxin"),      # Narrow therapeutic window
        (12, "Amiodarone"),  # Complex interactions
        (2, "Warfarin")      # High bleeding risk
    ]
    
    for med_id, med_name in high_risk_meds:
        scores = predictor.predict(cursor, complex_patient, med_id)
        print_risk_scores(f"Adding {med_name} to multiple medications", scores)

    # Test Case 5: Discontinued Medication Handling
    print("\n=== Testing Discontinued Medication Handling ===")
    cursor.execute("""
        SELECT DISTINCT patient_id 
        FROM Patient_Medications 
        WHERE dosage = 'DISCONTINUED'
        LIMIT 1
    """)
    discontinued_patient = cursor.fetchone()['patient_id']
    
    cursor.execute("""
        SELECT m.name, pm.dosage, pm.timestamp
        FROM Patient_Medications pm
        JOIN Medications m ON pm.medication_id = m.medication_id
        WHERE pm.patient_id = ?
        ORDER BY pm.timestamp DESC
    """, (discontinued_patient,))
    
    print("\nMedication History:")
    for med in cursor.fetchall():
        print(f"• {med['name']}: {med['dosage']} ({med['timestamp']})")
    
    scores = predictor.predict(cursor, discontinued_patient)
    print_risk_scores("Scores with discontinued medications", scores)

    conn.close()
    return "Testing completed"

if __name__ == "__main__":
    test_risk_predictor()