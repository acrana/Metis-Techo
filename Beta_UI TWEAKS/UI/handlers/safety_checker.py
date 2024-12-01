class SafetyChecker:
    # Class-level constants
    VITAL_CHECKS = {
        # Beta Blockers/Antiarrhythmics - concerned about low HR/BP
        'Metoprolol': {'heart_rate': 'low', 'blood_pressure_systolic': 'low'},
        'Amiodarone': {'heart_rate': 'low'},
        'Digoxin': {'heart_rate': 'low'},
        
        # ACE Inhibitors/Vasodilators - concerned about low BP
        'Lisinopril': {'blood_pressure_systolic': 'low'},
        'Nitroglycerin': {'blood_pressure_systolic': 'low'},
        'Sildenafil': {'blood_pressure_systolic': 'low'},
        
        # Diuretics - concerned about low BP
        'Furosemide': {'blood_pressure_systolic': 'low'},
        
        # Opioids - concerned about low RR and low BP
        'Morphine': {'respiratory_rate': 'low', 'blood_pressure_systolic': 'low'}
    }

    LAB_CHECKS = {
        # Cardiac Medications
        'Amiodarone': {'liver enzymes': 'high', 'qtc': 'high'},
        'Digoxin': {'potassium': 'both', 'digoxin level': 'high'},
        
        # ACE Inhibitors
        'Lisinopril': {'potassium': 'high', 'creatinine': 'high'},
        
        # Diuretics
        'Furosemide': {'potassium': 'low', 'creatinine': 'high'},
        
        # Antipsychotics
        'Seroquel': {'qtc': 'high', 'liver enzymes': 'high'},
        
        # Antibiotics
        'Vancomycin': {'creatinine': 'high'},
        
        # Blood thinners
        'Warfarin': {'inr': 'high'},
        
        # Antiemetics
        'Zofran': {'qtc': 'high'},
        
        # Statins
        'Atorvastatin': {'liver enzymes': 'high'},
        
        # Steroids
        'Hydrocortisone': {'blood glucose': 'high'},
        
        # Others
        'Insulin': {'Blood Glucose': 'low'}
    }
    def __init__(self, cdss, patient_id, medication):
        self.cdss = cdss
        self.patient_id = patient_id
        self.medication = medication
        
        # Get current and historical patient data
        self.current_vitals = self.cdss.get_recent_vitals(patient_id)
        self.historical_vitals = self.get_historical_vitals()
        self.current_labs = self.cdss.get_recent_labs(patient_id)
        self.historical_labs = self.get_historical_labs()

        # Get medication ID and assessment
        self.medication_id = self.get_medication_id()
        self.assessment = self.cdss.calculate_risk_on_medication_add(
            patient_id,
            self.medication_id
        ) if self.medication_id else None

    def get_medication_id(self):
        """Get medication ID from name"""
        cursor = self.cdss.conn.cursor()
        cursor.execute("SELECT medication_id FROM Medications WHERE name = ?", 
                      (self.medication,))
        result = cursor.fetchone()
        return result['medication_id'] if result else None

    def get_historical_vitals(self):
        """Get vital sign history for patient"""
        cursor = self.cdss.conn.cursor()
        cursor.execute("""
            SELECT * FROM Vitals 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC
        """, (self.patient_id,))
        return cursor.fetchall()

    def get_historical_labs(self):
        """Get lab result history for patient"""
        cursor = self.cdss.conn.cursor()
        cursor.execute("""
            SELECT lr.*, l.lab_name, lr.timestamp
            FROM Lab_Results lr
            JOIN Labs l ON lr.lab_id = l.lab_id
            WHERE lr.patient_id = ?
            ORDER BY lr.timestamp DESC
        """, (self.patient_id,))
        return cursor.fetchall()

    def get_vital_warnings(self):
        """Get warnings for current vital signs"""
        warnings = []
        if not self.current_vitals:
            return warnings

        vital_concerns = self.VITAL_CHECKS.get(self.medication, {})
        
        # Check heart rate if relevant
        if 'heart_rate' in vital_concerns and vital_concerns['heart_rate'] == 'low':
            hr = float(self.current_vitals['heart_rate'])
            min_normal = float(self.current_vitals['min_normal'])
            if hr < min_normal:
                warnings.append(f"• Heart Rate: {hr}")
        
        # Check blood pressure if relevant
        if 'blood_pressure_systolic' in vital_concerns and vital_concerns['blood_pressure_systolic'] == 'low':
            sys_bp = float(self.current_vitals['blood_pressure_systolic'])
            if sys_bp < 90:
                warnings.append(f"• Blood Pressure: {sys_bp}/{self.current_vitals['blood_pressure_diastolic']}")
        
        # Check respiratory rate if relevant
        if 'respiratory_rate' in vital_concerns and vital_concerns['respiratory_rate'] == 'low':
            rr = float(self.current_vitals['respiratory_rate'])
            if rr < 12:
                warnings.append(f"• Respiratory Rate: {rr}")

        return warnings

    def get_lab_warnings(self):
        """Get warnings for current lab results"""
        warnings = []
        lab_concerns = self.LAB_CHECKS.get(self.medication, {})
        
        for lab in self.current_labs:
            if lab['lab_name'].lower() in lab_concerns:
                try:
                    value = float(lab['result'])
                    min_normal = float(lab['min_normal'])
                    max_normal = float(lab['max_normal'])
                    concern = lab_concerns[lab['lab_name'].lower()]
                    
                    if concern == 'high' and value > max_normal:
                        warnings.append(f"• {lab['lab_name']}: {value}")
                    elif concern == 'low' and value < min_normal:
                        warnings.append(f"• {lab['lab_name']}: {value}")
                    elif concern == 'both' and (value < min_normal or value > max_normal):
                        warnings.append(f"• {lab['lab_name']}: {value}")
                except (ValueError, KeyError, TypeError):
                    continue
        
        return warnings

    def check_historical_vitals(self):
        """Check for historical vital sign issues"""
        warnings = []
        vital_concerns = self.VITAL_CHECKS.get(self.medication, {})
        
        for record in self.historical_vitals:
            # Check heart rate if relevant
            if 'heart_rate' in vital_concerns and vital_concerns['heart_rate'] == 'low':
                hr = float(record['heart_rate'])
                if hr < 60:
                    warnings.append(f"• Historical Heart Rate: {hr} on {record['timestamp'].split()[0]}")
            
            # Check blood pressure if relevant
            if 'blood_pressure_systolic' in vital_concerns and vital_concerns['blood_pressure_systolic'] == 'low':
                sys_bp = float(record['blood_pressure_systolic'])
                if sys_bp < 90:
                    warnings.append(f"• Historical Blood Pressure: {sys_bp}/{record['blood_pressure_diastolic']} on {record['timestamp'].split()[0]}")
            
            # Check respiratory rate if relevant
            if 'respiratory_rate' in vital_concerns and vital_concerns['respiratory_rate'] == 'low':
                rr = float(record['respiratory_rate'])
                if rr < 12:
                    warnings.append(f"• Historical Respiratory Rate: {rr} on {record['timestamp'].split()[0]}")

        return warnings

    def check_historical_labs(self):
        """Check for historical lab abnormalities"""
        warnings = []
        lab_concerns = self.LAB_CHECKS.get(self.medication, {})
        
        for lab in self.historical_labs:
            if lab['lab_name'].lower() in lab_concerns:
                try:
                    value = float(lab['result'])
                    cursor = self.cdss.conn.cursor()
                    cursor.execute("SELECT * FROM Lab_Ranges WHERE lab_name = ?", 
                                 (lab['lab_name'],))
                    ranges = cursor.fetchone()
                    
                    if ranges:
                        concern = lab_concerns[lab['lab_name'].lower()]
                        if concern == 'high' and value > float(ranges['max_normal']):
                            warnings.append(f"• Historical {lab['lab_name']}: {value} on {lab['timestamp'].split()[0]}")
                        elif concern == 'low' and value < float(ranges['min_normal']):
                            warnings.append(f"• Historical {lab['lab_name']}: {value} on {lab['timestamp'].split()[0]}")
                        elif concern == 'both' and (value < float(ranges['min_normal']) or value > float(ranges['max_normal'])):
                            warnings.append(f"• Historical {lab['lab_name']}: {value} on {lab['timestamp'].split()[0]}")
                except (ValueError, TypeError, KeyError):
                    continue

        return warnings

    def get_safety_warnings(self, frequency, duration):
        """Get complete safety assessment message"""
        msg = f"Please confirm medication order:\n\n"
        msg += f"Medication: {self.medication}\n"
        msg += f"Frequency: {frequency}\n"
        msg += f"Duration: {duration}\n\n"
        
        warnings = []
        
        # Add interaction warnings
        if self.assessment:
            seen_interactions = set()
            interactions = []
            for interaction in self.assessment['medication_interactions']:
                if interaction['interaction_type'].lower() == 'major':
                    pair_key = tuple(sorted([interaction['med1_name'], interaction['med2_name']]))
                    if pair_key not in seen_interactions:
                        seen_interactions.add(pair_key)
                        interactions.append(interaction)
                        
            if interactions:
                warnings.append("\nMajor Drug Interactions:")
                for interaction in interactions:
                    warnings.append(f"• {interaction['med1_name']} + {interaction['med2_name']}: "
                                  f"{interaction['description']}")

        # Add current vital sign warnings
        vital_warnings = self.get_vital_warnings()
        if vital_warnings:
            warnings.append(f"\nCurrent Abnormal Vital Signs:")
            warnings.extend(vital_warnings)

        # Add historical vital sign warnings
        historical_vital_warnings = self.check_historical_vitals()
        if historical_vital_warnings:
            warnings.append(f"\nHistory of Abnormal Vital Signs:")
            warnings.extend(historical_vital_warnings)

        # Add current lab warnings
        lab_warnings = self.get_lab_warnings()
        if lab_warnings:
            warnings.append(f"\nCurrent Abnormal Lab Values:")
            warnings.extend(lab_warnings)

        # Add historical lab warnings
        historical_lab_warnings = self.check_historical_labs()
        if historical_lab_warnings:
            warnings.append(f"\nHistory of Abnormal Lab Values:")
            warnings.extend(historical_lab_warnings)

        if warnings:
            msg += "\n⚠️ WARNING ⚠️\n"
            msg += "\n".join(warnings)
        else:
            msg += "\nNo critical safety concerns identified."

        return msg