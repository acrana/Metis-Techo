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
    
    def print_risk_scores(desc, scores, explanation=None):
        print(f"\n{desc}:")
        print(f"ADE Risk: {scores['ade_risk']:.1%}")
        print(f"Interaction Risk: {scores['interaction_risk']:.1%}")
        print(f"Overall Risk: {scores['overall_risk']:.1%}")
        if explanation:
            print("\nRisk Factors:")
            print(explanation)
    
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

    # Test Case 2: Single common medication with time-based risk assessment
    print("\n=== Testing Single Common Medication with Time Analysis ===")
    medications = [
        (3, "Metoprolol"),   # Common beta blocker
        (1, "Aspirin"),      # Common antiplatelet
        (14, "Lisinopril")   # Common ACE inhibitor
    ]
    
    for med_id, med_name in medications:
        scores = predictor.predict(cursor, clean_patient, med_id)
        explanation = predictor.explain_prediction(cursor, clean_patient, med_id)
        print_risk_scores(f"Single medication ({med_name})", scores, explanation)

    # Test Case 3: Patient with medication history and multiple ADEs
    print("\n=== Testing Patient with Complex ADE History ===")
    cursor.execute("""
        SELECT am.patient_id, COUNT(*) as ade_count
        FROM ADE_Monitoring am
        GROUP BY am.patient_id
        HAVING ade_count > 1
        ORDER BY ade_count DESC
        LIMIT 1
    """)
    ade_patient = cursor.fetchone()['patient_id']
    
    # Get their ADE history with timing analysis
    cursor.execute("""
        SELECT m.name, am.description, am.timestamp,
               julianday('now') - julianday(am.timestamp) as days_ago
        FROM ADE_Monitoring am
        JOIN Medications m ON am.medication_id = m.medication_id
        WHERE am.patient_id = ?
        ORDER BY am.timestamp DESC
    """, (ade_patient,))
    
    print("\nADE History:")
    for ade in cursor.fetchall():
        days_ago = round(float(ade['days_ago']))
        print(f"• {ade['days_ago']:.0f} days ago: {ade['name']} - {ade['description']}")
    
    scores = predictor.predict(cursor, ade_patient)
    explanation = predictor.explain_prediction(cursor, ade_patient, None)
    print_risk_scores("Base scores with ADE history", scores, explanation)

    # Test Case 4: Known high-risk combinations
    print("\n=== Testing Known High-Risk Combinations ===")
    # Find patient with Warfarin
    cursor.execute("""
        SELECT pm.patient_id
        FROM Patient_Medications pm
        WHERE pm.medication_id = 2  -- Warfarin
        AND pm.dosage != 'DISCONTINUED'
        LIMIT 1
    """)
    warfarin_patient = cursor.fetchone()['patient_id']
    
    # Test adding medications known to interact with Warfarin
    high_risk_combinations = [
        (1, "Aspirin"),       # Known bleeding risk with Warfarin
        (12, "Amiodarone"),   # Strong interaction with Warfarin
        (9, "Digoxin")        # Moderate interaction risk
    ]
    
    print("\nStarting with Warfarin patient")
    base_scores = predictor.predict(cursor, warfarin_patient)
    print_risk_scores("Baseline with Warfarin", base_scores)
    
    for med_id, med_name in high_risk_combinations:
        scores = predictor.predict(cursor, warfarin_patient, med_id)
        explanation = predictor.explain_prediction(cursor, warfarin_patient, med_id)
        print_risk_scores(f"Adding {med_name} to Warfarin", scores, explanation)

    # Test Case 5: Multiple Medication Changes
    print("\n=== Testing Impact of Medication Changes ===")
    cursor.execute("""
        SELECT pm.patient_id, COUNT(*) as change_count
        FROM Patient_Medications pm
        WHERE julianday('now') - julianday(pm.timestamp) <= 90
        GROUP BY pm.patient_id
        HAVING change_count >= 3
        ORDER BY change_count DESC
        LIMIT 1
    """)
    frequent_change_patient = cursor.fetchone()['patient_id']
    
    cursor.execute("""
        SELECT m.name, pm.dosage, pm.timestamp,
               julianday('now') - julianday(pm.timestamp) as days_ago
        FROM Patient_Medications pm
        JOIN Medications m ON pm.medication_id = m.medication_id
        WHERE pm.patient_id = ?
        ORDER BY pm.timestamp DESC
    """, (frequent_change_patient,))
    
    print("\nMedication Changes (Last 90 Days):")
    for med in cursor.fetchall():
        days_ago = round(float(med['days_ago']))
        if days_ago <= 90:
            print(f"• {days_ago} days ago: {med['name']} - {med['dosage']}")
    
    scores = predictor.predict(cursor, frequent_change_patient)
    explanation = predictor.explain_prediction(cursor, frequent_change_patient, None)
    print_risk_scores("Scores with frequent medication changes", scores, explanation)

    # Test Case 6: Testing time decay of risk factors
    print("\n=== Testing Risk Factor Time Decay ===")
    # Get patient with old and recent ADEs
    cursor.execute("""
        SELECT am.patient_id
        FROM ADE_Monitoring am
        GROUP BY am.patient_id
        HAVING MIN(julianday('now') - julianday(am.timestamp)) >= 180
        AND MAX(julianday('now') - julianday(am.timestamp)) <= 30
        LIMIT 1
    """)
    time_decay_patient = cursor.fetchone()['patient_id']
    
    cursor.execute("""
        SELECT m.name, am.description, am.timestamp,
               julianday('now') - julianday(am.timestamp) as days_ago
        FROM ADE_Monitoring am
        JOIN Medications m ON am.medication_id = m.medication_id
        WHERE am.patient_id = ?
        ORDER BY am.timestamp DESC
    """, (time_decay_patient,))
    
    print("\nADE History with Timing:")
    for ade in cursor.fetchall():
        days_ago = round(float(ade['days_ago']))
        print(f"• {days_ago} days ago: {ade['name']} - {ade['description']}")
    
    scores = predictor.predict(cursor, time_decay_patient)
    explanation = predictor.explain_prediction(cursor, time_decay_patient, None)
    print_risk_scores("Scores showing time decay", scores, explanation)

    conn.close()
    return "Testing completed successfully"

if __name__ == "__main__":
    test_risk_predictor()
