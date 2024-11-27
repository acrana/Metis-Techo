import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRiskPredictor:
    def __init__(self, cdss_system, config: Dict[str, Any] = None):
        self.cdss = cdss_system
        
        # Default configuration
        default_config = {
            'lookback_days': 1000,  # Increased lookback
            'weights': {
                'historical_ade': 0.35,  
                'vital_trends': 0.25,    
                'lab_trends': 0.40  # Increased because labs show clear issues
            },
            'decay_factors': {
                'ade': 365,  # ADEs remain relevant for longer
                'vitals': 30,      
                'labs': 60
            },
            'medication_vital_concerns': {
                'Metoprolol': ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic'],
                'Lisinopril': ['blood_pressure_systolic', 'blood_pressure_diastolic'],
                'Amiodarone': ['heart_rate', 'blood_pressure_systolic'],
                'Seroquel': ['heart_rate', 'blood_pressure_systolic'],
                'Warfarin': []
            },
            'medication_lab_concerns': {
                'Metoprolol': ['potassium', 'creatinine'],
                'Lisinopril': ['potassium', 'creatinine'],
                'Amiodarone': ['liver enzymes', 'thyroid function'],
                'Warfarin': ['inr', 'hemoglobin'],
                'Seroquel': ['blood glucose', 'liver enzymes']
            }
        }
        
        # Update with custom config if provided
        if config:
            default_config.update(config)
        
        self.config = default_config

    def calculate_time_weight(self, days_ago: float, category: str) -> float:
        """Calculate time-based weight using exponential decay"""
        half_life = self.config['decay_factors'][category]
        decay_constant = np.log(2) / half_life
        return np.exp(-decay_constant * days_ago)

    def analyze_historical_ades(self, patient_id: str, medication: str) -> Tuple[float, List[Dict]]:
        """Analyze historical ADEs with sophisticated pattern matching"""
        cursor = self.cdss.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=self.config['lookback_days'])).isoformat()
        
        # Get related medications
        cursor.execute("""
            SELECT m1.type as med_type, m2.medication_id, m2.name
            FROM Medications m1
            JOIN Medications m2 ON m1.type = m2.type
            WHERE m1.name = ?
        """, (medication,))
        related_meds = cursor.fetchall()
        med_class = related_meds[0]['med_type'] if related_meds else None
        related_med_ids = [row['medication_id'] for row in related_meds]

        # Mechanism-related terms
        mechanism_related = {
            'Metoprolol': ['bradycardia', 'hypotension', 'heart rate', 'blood pressure'],
            'Amiodarone': ['bradycardia', 'qtc', 'prolongation', 'heart rate'],
            'Lisinopril': ['hypotension', 'blood pressure', 'kidney'],
            'Warfarin': ['bleeding', 'inr', 'coagulation'],
            'Seroquel': ['qtc', 'blood sugar', 'glucose']
        }

        # Query historical ADEs
        cursor.execute("""
            SELECT 
                ade.*,
                m.name as med_name,
                m.type as med_type
            FROM ADE_Monitoring ade
            JOIN Medications m ON ade.medication_id = m.medication_id
            WHERE ade.patient_id = ?
            AND ade.timestamp >= ?
            ORDER BY ade.timestamp DESC
        """, (patient_id, cutoff_date))
        
        ade_history = cursor.fetchall()
        
        if not ade_history:
            return 0.0, []
        
        risk_score = 0.0
        relevant_ades = []
        
        related_terms = mechanism_related.get(medication, [])
        
        for ade in ade_history:
            days_ago = (datetime.now() - datetime.fromisoformat(ade['timestamp'])).days
            time_weight = self.calculate_time_weight(days_ago, 'ade')
            
            desc_lower = ade['description'].lower()
            mechanism_match = any(term in desc_lower for term in related_terms)
            
            if ade['med_name'] == medication:
                relevance = 1.0  # Exact same medication
            elif mechanism_match:
                relevance = 0.9  # Related mechanism of action
            elif ade['med_type'] == med_class:
                relevance = 0.8  # Same medication class
            elif ade['medication_id'] in related_med_ids:
                relevance = 0.6  # Related medication
            else:
                relevance = 0.2  # Other medication
            
            severity = 1.0
            if any(term in desc_lower for term in ['severe', 'critical', 'emergency']):
                severity = 2.0
            elif any(term in desc_lower for term in ['moderate', 'significant']):
                severity = 1.5
            
            if not ade['resolved']:
                severity *= 1.5
            
            event_score = time_weight * relevance * severity
            risk_score = max(risk_score, event_score)
            
            if event_score > 0.2:
                relevant_ades.append({
                    'timestamp': ade['timestamp'],
                    'medication': ade['med_name'],
                    'description': ade['description'],
                    'score': event_score,
                    'resolved': bool(ade['resolved'])
                })
        
        return risk_score, sorted(relevant_ades, key=lambda x: x['score'], reverse=True)

    def analyze_vital_trends(self, patient_id: str, medication: str) -> Tuple[float, Dict]:
        """Analyze vital sign trends focusing on medication-specific concerns"""
        cursor = self.cdss.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=90)).isoformat()
        
        vital_concerns = self.config['medication_vital_concerns'].get(medication, [])
        
        if not vital_concerns:
            return 0.0, {}
        
        cursor.execute("""
            SELECT *
            FROM Vitals
            WHERE patient_id = ?
            AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (patient_id, cutoff_date))
        vital_history = cursor.fetchall()
        
        if not vital_history:
            return 0.0, {}
        
        cursor.execute("SELECT * FROM Vital_Ranges")
        vital_ranges = {row['vital_name']: dict(row) for row in cursor.fetchall()}
        
        trends = defaultdict(list)
        risk_scores = defaultdict(float)
        
        for vital in vital_history:
            timestamp = datetime.fromisoformat(vital['timestamp'])
            days_ago = (datetime.now() - timestamp).days
            time_weight = self.calculate_time_weight(days_ago, 'vitals')
            
            for param in vital_concerns:
                if vital[param] is not None:
                    value = float(vital[param])
                    ranges = vital_ranges.get(param)
                    
                    if ranges:
                        critical_min = float(ranges['critical_min'])
                        critical_max = float(ranges['critical_max'])
                        min_normal = float(ranges['min_normal'])
                        max_normal = float(ranges['max_normal'])
                        
                        if value <= critical_min or value >= critical_max:
                            deviation = 1.0
                            severity_factor = 2.0
                        else:
                            normal_range = max_normal - min_normal
                            if normal_range > 0:
                                if value < min_normal:
                                    deviation = (min_normal - value) / (min_normal - critical_min)
                                elif value > max_normal:
                                    deviation = (value - max_normal) / (critical_max - max_normal)
                                else:
                                    deviation = 0
                                severity_factor = 1.0
                            else:
                                deviation = 0
                                severity_factor = 1.0
                        
                        weighted_score = deviation * time_weight * severity_factor
                        risk_scores[param] = max(risk_scores[param], weighted_score)
                        
                        trends[param].append({
                            'timestamp': vital['timestamp'],
                            'value': value,
                            'deviation': weighted_score
                        })
        
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        
        return max_risk, dict(trends)

    def analyze_lab_trends(self, patient_id: str, medication: str) -> Tuple[float, Dict]:
        """Analyze lab value trends with sophisticated pattern matching"""
        cursor = self.cdss.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=180)).isoformat()
        
        lab_concerns = self.config['medication_lab_concerns'].get(medication, [])
        
        if not lab_concerns:
            return 0.0, {}
        
        placeholders = ','.join('?' for _ in lab_concerns)
        cursor.execute(f"""
            SELECT lr.*, l.lab_name
            FROM Lab_Results lr
            JOIN Labs l ON lr.lab_id = l.lab_id
            WHERE lr.patient_id = ?
            AND lr.timestamp >= ?
            AND LOWER(l.lab_name) IN ({placeholders})
            ORDER BY lr.timestamp DESC
        """, [patient_id, cutoff_date] + lab_concerns)
        
        lab_history = cursor.fetchall()
        
        if not lab_history:
            return 0.0, {}
        
        cursor.execute("SELECT * FROM Lab_Ranges")
        lab_ranges = {row['lab_name'].lower(): dict(row) for row in cursor.fetchall()}
        
        trends = defaultdict(list)
        risk_scores = defaultdict(float)
        
        for lab in lab_history:
            try:
                value = float(lab['result'])
                timestamp = datetime.fromisoformat(lab['timestamp'])
                days_ago = (datetime.now() - timestamp).days
                time_weight = self.calculate_time_weight(days_ago, 'labs')
                
                ranges = lab_ranges.get(lab['lab_name'].lower())
                if ranges:
                    critical_min = float(ranges['critical_min'])
                    critical_max = float(ranges['critical_max'])
                    min_normal = float(ranges['min_normal'])
                    max_normal = float(ranges['max_normal'])
                    
                    if value <= critical_min or value >= critical_max:
                        deviation = 1.0
                        severity_factor = 2.0
                    else:
                        normal_range = max_normal - min_normal
                        if normal_range > 0:
                            if value < min_normal:
                                deviation = (min_normal - value) / (min_normal - critical_min)
                            elif value > max_normal:
                                deviation = (value - max_normal) / (critical_max - max_normal)
                            else:
                                deviation = 0
                            severity_factor = 1.0
                        else:
                            deviation = 0
                            severity_factor = 1.0
                    
                    weighted_score = deviation * time_weight * severity_factor
                    risk_scores[lab['lab_name']] = max(
                        risk_scores[lab['lab_name']],
                        weighted_score
                    )
                    
                    trends[lab['lab_name']].append({
                        'timestamp': lab['timestamp'],
                        'value': value,
                        'deviation': weighted_score
                    })
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing lab value: {e}")
                continue
        
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        
        return max_risk, dict(trends)

    def calculate_combined_risk(self, patient_id: str, medication: str) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment with detailed breakdown"""
        
        ade_score, ade_details = self.analyze_historical_ades(patient_id, medication)
        vital_score, vital_trends = self.analyze_vital_trends(patient_id, medication)
        lab_score, lab_trends = self.analyze_lab_trends(patient_id, medication)
        
        total_risk = (
            ade_score * self.config['weights']['historical_ade'] +
            vital_score * self.config['weights']['vital_trends'] +
            lab_score * self.config['weights']['lab_trends']
        )
        
        assessment = {
            'total_risk_score': total_risk,
            'risk_level': self._get_risk_level(total_risk),
            'components': {
                'historical_ades': {
                    'score': ade_score,
                    'weight': self.config['weights']['historical_ade'],
                    'details': ade_details
                },
                'vital_trends': {
                    'score': vital_score,
                    'weight': self.config['weights']['vital_trends'],
                    'trends': vital_trends
                },
                'lab_trends': {
                    'score': lab_score,
                    'weight': self.config['weights']['lab_trends'],
                    'trends': lab_trends
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return assessment

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical risk level"""
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MODERATE"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def get_risk_factors(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key risk factors from assessment"""
        risk_factors = []
        
        for ade in assessment['components']['historical_ades']['details']:
            if ade['score'] > 0.3:
                risk_factors.append({
                    'type': 'Historical ADE',
                    'severity': 'High' if ade['score'] > 0.7 else 'Moderate',
                    'description': ade['description'],
                    'timestamp': ade['timestamp']
                })
        
        vital_trends = assessment['components']['vital_trends']['trends']
        for vital, readings in vital_trends.items():
            high_deviations = [r for r in readings if r['deviation'] > 0.5]
            if high_deviations:
                risk_factors.append({
                    'type': 'Vital Sign Trend',
                    'severity': 'High' if any(r['deviation'] > 0.7 for r in high_deviations) else 'Moderate',
                    'description': f"Abnormal {vital} readings",
                    'details': f"{len(high_deviations)} concerning readings"
                })
        
        lab_trends = assessment['components']['lab_trends']['trends']
        for lab, readings in lab_trends.items():
            high_deviations = [r for r in readings if r['deviation'] > 0.5]
            if high_deviations:
                risk_factors.append({
                    'type': 'Lab Value Trend',
                    'severity': 'High' if any(r['deviation'] > 0.7 for r in high_deviations) else 'Moderate',
                    'description': f"Abnormal {lab} values",
                    'details': f"{len(high_deviations)} concerning results"
                })
        
        return risk_factors
