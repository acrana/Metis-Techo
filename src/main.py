from tkinter import messagebox
from patients import patients
from medications import medications

# Function to check contraindications with GUI notifications
def check_contraindications_gui(patient, medication_name):
    if medication_name not in medications:
        messagebox.showerror("Error", f"Medication '{medication_name}' not found.")
        return False

    med_info = medications[medication_name]
    contraindications = med_info.get('Contraindications', {})

    # Check for contraindications
    for param, condition in contraindications.items():
        if param in patient['LabResults']:
            patient_value = patient['LabResults'][param]
            if condition(patient_value):
                # Show a warning and ask the user if they want to proceed
                proceed = messagebox.askyesno("Contraindication Alert",
                                              f"{medication_name} is contraindicated due to {param} level of {patient_value}.\nDo you want to proceed?")
                if not proceed:
                    return False  # Stop the prescription if the user selects "No"
    return True  # Proceed if no contraindications or user agrees

# Function to prescribe medication with GUI notifications
def prescribe_and_monitor_gui(patient, medication_name):
    if not check_contraindications_gui(patient, medication_name):
        # Medication can't be prescribed due to contraindications
        messagebox.showwarning("Warning", f"Cannot prescribe '{medication_name}' due to contraindications.")
        return False

    if medication_name in patient['Medications']:
        # Medication is already prescribed
        messagebox.showinfo("Info", f"Patient {patient['PatientID']} is already taking '{medication_name}'.")
        return False

    # Prescribe the medication
    patient['Medications'].append(medication_name)
    return True

# Function to update patient lab results and issue warnings
def update_patient_labs(patient, new_lab_results):
    # Update the patient's lab results with the new values
    patient['LabResults'].update(new_lab_results)

    # Recheck the patient's medications for any warnings due to updated lab results
    for medication in patient['Medications']:
        med_info = medications[medication]
        monitoring_params = med_info.get('MonitoringParameters', [])

        for param in monitoring_params:
            if param in new_lab_results:
                patient_value = new_lab_results[param]
                # Check if the updated lab value exceeds the safe limit
                if param == 'Creatinine' and patient_value > 2.0:
                    messagebox.showwarning("Warning", f"Patient {patient['PatientID']} has elevated Creatinine ({patient_value}) while on {medication}. Please proceed with caution.")
                # Add more conditions for other monitoring parameters as needed

# Function to monitor patient labs continuously
def monitor_patient_labs(patient, medication_name):
    med_info = medications[medication_name]
    monitoring_params = med_info.get('MonitoringParameters', [])

    for param in monitoring_params:
        if param in patient['LabResults']:
            patient_value = patient['LabResults'][param]
            # Check if the patient's value is now beyond safe limits
            if param == 'QTc' and patient_value > 470:
                messagebox.showwarning("Warning", f"Patient {patient['PatientID']} has a high QTc ({patient_value}) while on {medication_name}.")
            if param == 'Creatinine' and patient_value > 2.0:
                messagebox.showwarning("Warning", f"Patient {patient['PatientID']} has elevated Creatinine ({patient_value}) while on {medication_name}.")
        else:
            messagebox.showwarning("Monitoring Warning", f"{param} is missing in the patient data for Patient {patient['PatientID']}.")

# Function to find a patient by ID
def find_patient(patient_id):
    for patient in patients:
        if patient['PatientID'] == patient_id:
            return patient
    return None

# Function to display patient information
def display_patient_info(patient):
    info = f"Patient ID: {patient['PatientID']}\n"
    info += f"Name: {patient['FirstName']} {patient['LastName']}\n"
    info += f"Age: {patient['Age']}\n"
    info += f"Gender: {patient['Gender']}\n"
    info += f"Allergies: {', '.join(patient.get('Allergies', [])) if patient.get('Allergies') else 'None'}\n"
   


if __name__ == '__main__':
    main()
