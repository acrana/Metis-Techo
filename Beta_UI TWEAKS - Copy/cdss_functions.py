import sqlite3
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

class CDSSSystem:
    def __init__(self, db_path: str = 'ai_cdss.db'):
        self.db_path = db_path
        self.conn = None
        self.connect()

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_patient_conditions(self, patient_id: str) -> List[Tuple[str, int]]:
        """Get all conditions for a patient"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.condition_name, c.condition_id
            FROM Patient_Conditions pc
            JOIN Conditions c ON pc.condition_id = c.condition_id
            WHERE pc.patient_id = ?
        """, (patient_id,))
        return [(row['condition_name'], row['condition_id']) for row in cursor.fetchall()]

    def get_patient_medications(self, patient_id: str) -> List[Tuple[int, str]]:
        """Get current medications for a patient"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT m.medication_id, m.name
            FROM Patient_Medications pm
            JOIN Medications m ON pm.medication_id = m.medication_id
            WHERE pm.patient_id = ?
            AND pm.timestamp = (
                SELECT MAX(timestamp)
                FROM Patient_Medications
                WHERE medication_id = pm.medication_id
                AND patient_id = pm.patient_id
            )
            AND pm.dosage != 'DISCONTINUED'
        """, (patient_id,))
        return [(row['medication_id'], row['name']) for row in cursor.fetchall()]

    def get_recent_vitals(self, patient_id: str) -> Dict[str, Any]:
        """Get most recent vitals for a patient"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT v.*, vr.min_normal, vr.max_normal, vr.critical_min, vr.critical_max
            FROM Vitals v
            LEFT JOIN Vital_Ranges vr ON vr.vital_name = 'heart_rate'
            WHERE v.patient_id = ?
            ORDER BY v.timestamp DESC
            LIMIT 1
        """, (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_recent_labs(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get recent lab results with their normal ranges"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT lr.*, l.lab_name, lrng.min_normal, lrng.max_normal, 
                   lrng.critical_min, lrng.critical_max
            FROM Lab_Results lr
            JOIN Labs l ON lr.lab_id = l.lab_id
            LEFT JOIN Lab_Ranges lrng ON l.lab_name = lrng.lab_name
            WHERE lr.patient_id = ?
            AND lr.timestamp = (
                SELECT MAX(timestamp)
                FROM Lab_Results
                WHERE lab_id = lr.lab_id
                AND patient_id = lr.patient_id
            )
        """, (patient_id,))
        return [dict(row) for row in cursor.fetchall()]

    def check_medication_interactions(self, medications: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """Check for interactions between medications"""
        interactions = []
        cursor = self.conn.cursor()
        
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                cursor.execute("""
                    SELECT mi.*, m1.name as med1_name, m2.name as med2_name
                    FROM Medication_Interactions mi
                    JOIN Medications m1 ON mi.medication_1_id = m1.medication_id
                    JOIN Medications m2 ON mi.medication_2_id = m2.medication_id
                    WHERE (medication_1_id = ? AND medication_2_id = ?)
                    OR (medication_1_id = ? AND medication_2_id = ?)
                """, (med1[0], med2[0], med2[0], med1[0]))
                
                row = cursor.fetchone()
                if row:
                    interactions.append(dict(row))
        
        return interactions

    def check_contraindications(self, 
                              conditions: List[Tuple[str, int]], 
                              medication_id: int) -> List[Dict[str, Any]]:
        """Check for contraindications between conditions and medication"""
        cursor = self.conn.cursor()
        contraindications = []
        
        for _, condition_id in conditions:
            cursor.execute("""
                SELECT cc.*, c.condition_name, m.name as medication_name
                FROM Condition_Contraindications cc
                JOIN Conditions c ON cc.condition_id = c.condition_id
                JOIN Medications m ON cc.medication_id = m.medication_id
                WHERE cc.condition_id = ? AND cc.medication_id = ?
            """, (condition_id, medication_id))
            
            row = cursor.fetchone()
            if row:
                contraindications.append(dict(row))
        
        return contraindications

    def check_vital_alerts(self, vitals: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check vitals against normal ranges"""
        alerts = []
        cursor = self.conn.cursor()
        
        # Get all vital ranges from the database
        cursor.execute("SELECT * FROM Vital_Ranges")
        vital_ranges = {row['vital_name']: dict(row) for row in cursor.fetchall()}
        
        vital_checks = [
            ('heart_rate', 'Heart Rate', 'bpm'),
            ('blood_pressure_systolic', 'Systolic BP', 'mmHg'),
            ('blood_pressure_diastolic', 'Diastolic BP', 'mmHg'),
            ('respiratory_rate', 'Respiratory Rate', '/min'),
            ('oxygen_saturation', 'O2 Saturation', '%'),
            ('temperature', 'Temperature', '°C')
        ]
        
        for vital_key, vital_name, unit in vital_checks:
            value = vitals.get(vital_key)
            if value is None:
                continue
                
            # Ensure we're working with float values
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
                
            ranges = vital_ranges.get(vital_key)
            if ranges is None:
                continue

            min_normal = float(ranges['min_normal'])
            max_normal = float(ranges['max_normal'])
            critical_min = float(ranges['critical_min'])
            critical_max = float(ranges['critical_max'])
            
            # Check ranges
            if value <= critical_min:
                alerts.append({
                    'severity': 'Critical',
                    'message': f'{vital_name} critically low: {value}'
                })
            elif value >= critical_max:
                alerts.append({
                    'severity': 'Critical',
                    'message': f'{vital_name} critically high: {value}'
                })
            elif value < min_normal:
                alerts.append({
                    'severity': 'Warning',
                    'message': f'{vital_name} below normal: {value}'
                })
            elif value > max_normal:
                alerts.append({
                    'severity': 'Warning',
                    'message': f'{vital_name} above normal: {value}'
                })
            
        return alerts

    def check_lab_alerts(self, labs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Check lab results against normal ranges"""
        alerts = []
        
        for lab in labs:
            try:
                value = float(lab['result'])
                lab_name = lab['lab_name']
                
                if value < float(lab['critical_min']):
                    alerts.append({
                        'severity': 'Critical',
                        'message': f'{lab_name} critically low: {value}'
                    })
                elif value > float(lab['critical_max']):
                    alerts.append({
                        'severity': 'Critical',
                        'message': f'{lab_name} critically high: {value}'
                    })
                elif value < float(lab['min_normal']):
                    alerts.append({
                        'severity': 'Warning',
                        'message': f'{lab_name} below normal: {value}'
                    })
                elif value > float(lab['max_normal']):
                    alerts.append({
                        'severity': 'Warning',
                        'message': f'{lab_name} above normal: {value}'
                    })
            except (ValueError, KeyError, TypeError):
                continue
        
        return alerts

    def calculate_risk_on_medication_add(self, patient_id: str, new_medication_id: int) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment for adding a new medication"""
        # Get patient's current conditions and medications
        conditions = self.get_patient_conditions(patient_id)
        current_medications = self.get_patient_medications(patient_id)
        
        # Add new medication to check interactions
        cursor = self.conn.cursor()
        cursor.execute("SELECT medication_id, name FROM Medications WHERE medication_id = ?", 
                      (new_medication_id,))
        new_med = cursor.fetchone()
        if new_med:
            medications_to_check = current_medications + [(new_med['medication_id'], new_med['name'])]
        else:
            medications_to_check = current_medications

        # Get patient's recent vitals and labs
        vitals = self.get_recent_vitals(patient_id)
        labs = self.get_recent_labs(patient_id)

        # Perform all checks
        interaction_alerts = self.check_medication_interactions(medications_to_check)
        contraindication_alerts = self.check_contraindications(conditions, new_medication_id)
        vital_alerts = self.check_vital_alerts(vitals)
        lab_alerts = self.check_lab_alerts(labs)

        # Compile alerts
        assessment = {
            'medication_interactions': interaction_alerts,
            'contraindications': contraindication_alerts,
            'vital_alerts': vital_alerts,
            'lab_alerts': lab_alerts,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_interactions': len(interaction_alerts),
                'total_contraindications': len(contraindication_alerts),
                'total_vital_alerts': len(vital_alerts),
                'total_lab_alerts': len(lab_alerts)
            }
        }

        # Log the assessment
        self.log_assessment(patient_id, new_medication_id, assessment)
        
        return assessment

    def log_assessment(self, patient_id: str, medication_id: int, assessment: Dict[str, Any]):
        """Log the risk assessment to the Alerts table"""
        cursor = self.conn.cursor()
        
        # Log critical alerts
        for alert_type, alerts in assessment.items():
            if isinstance(alerts, list):
                for alert in alerts:
                    if isinstance(alert, dict) and alert.get('severity') == 'Critical':
                        cursor.execute("""
                            INSERT INTO Alerts 
                            (patient_id, timestamp, alert_type, alert_message, urgency_level, 
                             acknowledged, resolved)
                            VALUES (?, ?, ?, ?, 'Critical', 0, 0)
                        """, (
                            patient_id,
                            datetime.now().isoformat(),
                            alert_type,
                            alert.get('message', 'No message provided')
                        ))
        
        self.conn.commit()