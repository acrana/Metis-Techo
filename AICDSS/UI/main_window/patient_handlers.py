class PatientHandlersMixin:
    def load_patients(self):
        
        cursor = self.cdss.conn.cursor()
        cursor.execute("""
            SELECT patient_id, name, age, gender 
            FROM Patients 
            ORDER BY name
        """)
        self.patients = cursor.fetchall()
        self.patient_combo.addItems(
            [f"{p['name']} ({p['patient_id']})" for p in self.patients]
        )

    def filter_patients(self, text):
        
        search_text = text.lower()
        self.patient_combo.clear()
        
        filtered_patients = [
            f"{p['name']} ({p['patient_id']})"
            for p in self.patients
            if search_text in p['name'].lower() or 
               search_text in p['patient_id'].lower()
        ]
        
        self.patient_combo.addItems(filtered_patients)

    def on_patient_select(self, index):
        
        if index < 0:
            return
            
        selection = self.patient_combo.currentText()
        self.current_patient_id = selection.split('(')[1].rstrip(')')
        print(f"Patient selected: {self.current_patient_id}")
        
        # Update all patient information
        self.update_patient_info()
        self.update_conditions()
        self.update_current_medications()
        self.update_vitals()
        
        # Enable medication selection in medications tab
        if hasattr(self, 'medications_tab'):
            self.medications_tab.med_combo.setEnabled(True)
        if hasattr(self, 'analytics_tab'):
            self.analytics_tab.update_analytics(self.current_patient_id)

    def update_patient_info(self):
        
        if not self.current_patient_id:
            return
            
        cursor = self.cdss.conn.cursor()
        cursor.execute("""
            SELECT p.*, GROUP_CONCAT(c.condition_name) as conditions
            FROM Patients p
            LEFT JOIN Patient_Conditions pc ON p.patient_id = pc.patient_id
            LEFT JOIN Conditions c ON pc.condition_id = c.condition_id
            WHERE p.patient_id = ?
            GROUP BY p.patient_id
        """, (self.current_patient_id,))
        patient = cursor.fetchone()
        
        if patient:
            # Update header info
            self.patient_info.setText(
                f"{patient['name']} | Age: {patient['age']} | {patient['gender']}"
            )
            
            # Update overview tab
            if hasattr(self, 'overview_tab'):
                self.overview_tab.update_patient_info(dict(patient))

    def update_conditions(self):
        
        conditions = self.cdss.get_patient_conditions(self.current_patient_id)
        if hasattr(self, 'overview_tab'):
            self.overview_tab.update_conditions(conditions)

    def refresh_vitals(self):
        
        if self.current_patient_id:
            self.update_vitals()

    def update_vitals(self):
        
        if not self.current_patient_id:
            return
            
        vitals = self.cdss.get_recent_vitals(self.current_patient_id)
        if hasattr(self, 'overview_tab'):
            self.overview_tab.update_vitals(vitals, self.vital_ranges)

    def update_current_medications(self):
        
        medications = self.get_formatted_medications()
        if hasattr(self, 'overview_tab'):
            self.overview_tab.update_medications(medications)
        if hasattr(self, 'medications_tab'):
            self.medications_tab.update_current_medications(medications)

    def get_formatted_medications(self):
        
        medications = []
        if not self.current_patient_id:
            return medications

        cursor = self.cdss.conn.cursor()
        current_meds = self.cdss.get_patient_medications(self.current_patient_id)
        
        for med_id, med_name in current_meds:
            medications.append({
                'name': med_name,
                'medication_id': med_id
            })
        
        return medications

    def load_ranges(self):
        
        cursor = self.cdss.conn.cursor()
        
        # Load vital ranges
        cursor.execute("SELECT * FROM Vital_Ranges")
        self.vital_ranges = {
            row['vital_name']: dict(row) 
            for row in cursor.fetchall()
        }
        
        # Load lab ranges
        cursor.execute("SELECT * FROM Lab_Ranges")
        self.lab_ranges = {
            row['lab_name']: dict(row) 
            for row in cursor.fetchall()
        }