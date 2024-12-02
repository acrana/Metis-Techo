from PySide6.QtWidgets import QMessageBox
from datetime import datetime
from .safety_checker import SafetyChecker

class MedicationHandlersMixin:
    

    def load_medications(self):
        
        cursor = self.cdss.conn.cursor()
        cursor.execute("""
            SELECT medication_id, name, type, route, 
                   dosage_min, dosage_max, unit 
            FROM Medications 
            ORDER BY name
        """)
        self.medications_data = cursor.fetchall()
        if hasattr(self, 'medications_tab'):
            self.medications_tab.med_combo.clear()
            self.medications_tab.med_combo.addItems(
                [med['name'] for med in self.medications_data]
            )

    def on_medication_select(self, index):
        
        if index < 0:
            return

        medication = self.medications_tab.med_combo.currentText()
        self.current_medication = medication
        
        # Enable other fields
        self.medications_tab.freq_combo.setEnabled(True)
        self.medications_tab.duration_combo.setEnabled(True)
        self.medications_tab.order_button.setEnabled(True)

        # Update safety information
        self.update_safety_check()

    def update_safety_check(self):
        
        if not self.current_patient_id or not self.current_medication:
            return

        medication_id = next(
            (med['medication_id'] for med in self.medications_data
             if med['name'] == self.current_medication),
            None
        )

        if not medication_id:
            return

        try:
            # Get current medications for interaction check
            current_meds = self.cdss.get_patient_medications(self.current_patient_id)
            
            # Add selected medication to check interactions
            current_meds.append((medication_id, self.current_medication))
            
            # Check interactions
            interactions = self.cdss.check_medication_interactions(current_meds)
            self.medications_tab.update_interactions_table(interactions)
            
            # Check contraindications
            conditions = self.cdss.get_patient_conditions(self.current_patient_id)
            contraindications = self.cdss.check_contraindications(conditions, medication_id)
            self.medications_tab.update_contraindications_table(contraindications)

            # Get risk assessment
            assessment = self.risk_predictor.calculate_combined_risk(
                self.current_patient_id,
                self.current_medication
            )
            self.medications_tab.update_risk_display(assessment)

        except Exception as e:
            print(f"Error in update_safety_check: {str(e)}")
            QMessageBox.warning(
                self,
                "Safety Check Error",
                f"Error performing safety check: {str(e)}"
            )

    def place_order(self):
        selected_index = self.medications_tab.med_combo.currentIndex()
        if selected_index < 0:
            QMessageBox.warning(self, "Error", "Please select a medication")
            return

        medication = self.medications_data[selected_index]
        frequency = self.medications_tab.freq_combo.currentText()
        duration = self.medications_tab.duration_combo.currentText()
            
        if not frequency or not duration:
            QMessageBox.warning(self, "Error", "Please select frequency and duration")
            return

        try:
            # Get risk assessment
            assessment = self.risk_predictor.calculate_combined_risk(
                self.current_patient_id,
                medication['name']  # Use medication name from selected data
            )

            if assessment['risk_level'] in ['HIGH', 'MODERATE']:
                warning_msg = (
                    f"Risk Level: {assessment['risk_level']}\n"
                    f"Risk Score: {assessment['total_risk_score']:.3f}\n\n"
                )

                # Add ADE warnings
                ade_details = assessment['components']['historical_ades']
                if ade_details['details']:
                    warning_msg += "\nHistorical ADEs:\n"
                    for ade in ade_details['details']:
                        warning_msg += f"• {ade['description']}\n"

                # Add lab warnings
                lab_details = assessment['components']['lab_trends']
                if lab_details['trends']:
                    warning_msg += "\nAbnormal Lab Values:\n"
                    for lab, readings in lab_details['trends'].items():
                        for reading in readings[:1]:
                            warning_msg += f"• {lab}: {reading['value']}\n"

                warning_msg += "\nDo you want to proceed with the order?"

                reply = QMessageBox.warning(
                    self,
                    "High Risk Medication Order",
                    warning_msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.No:
                    return

            # Record the order - use medication_id directly from selected data
            cursor = self.cdss.conn.cursor()
            cursor.execute("""
                INSERT INTO Patient_Medications 
                (patient_id, medication_id, timestamp, dosage)
                VALUES (?, ?, ?, ?)
            """, (
                self.current_patient_id,
                medication['medication_id'],  # Use medication_id directly
                datetime.now().isoformat(),
                f"{frequency} - {duration}"
            ))

            self.cdss.conn.commit()
            QMessageBox.information(self, "Success", "Medication order placed successfully.")
            self.update_current_medications()

        except Exception as e:
            print(f"Error in place_order: {str(e)}")
            self.cdss.conn.rollback()
            QMessageBox.critical(self, "Error", f"Failed to place medication order: {str(e)}")

    def discontinue_medication(self, med):
        
        if not self.current_patient_id:
            return
            
        try:
            cursor = self.cdss.conn.cursor()
            
            # Add discontinuation record
            cursor.execute("""
                INSERT INTO Patient_Medications 
                (patient_id, medication_id, timestamp, dosage)
                VALUES (?, ?, ?, 'DISCONTINUED')
            """, (
                self.current_patient_id,
                med['medication_id'],
                datetime.now().isoformat()
            ))
            
            self.cdss.conn.commit()
            
            # Update displays
            self.update_current_medications()
            
            QMessageBox.information(
                self,
                "Success",
                f"Discontinued {med['name']}"
            )
            
        except Exception as e:
            print(f"Error in discontinue_medication: {str(e)}")  # Debug print
            self.cdss.conn.rollback()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to discontinue medication: {str(e)}"
            )

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