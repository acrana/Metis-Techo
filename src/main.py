# main.py

from patients import patients
from medications import medications, upper_limits_normal, lower_limits_normal

def check_contraindications(patient, medication_name):
    if medication_name not in medications:
        print(f"Medication '{medication_name}' not found.")
        return False

    med_info = medications[medication_name]
    contraindications = med_info.get('Contraindications', {})
    for param, condition in contraindications.items():
        if param == 'Allergies':
            # Check for allergy contraindications
            patient_allergies = patient.get('Allergies', [])
            if any(allergy in condition for allergy in patient_allergies):
                print(f"Contraindication Alert: Patient is allergic to {medication_name} or related drugs.")
                return False
        elif param == 'Gastrointestinal Issues':
            # Check for GI issues in medical history
            if 'Gastrointestinal Issues' in patient.get('MedicalHistory', []):
                print(f"Contraindication Alert: {medication_name} is contraindicated due to gastrointestinal issues.")
                return False
        else:
            # Lab value contraindications
            patient_value = patient['LabResults'].get(param)
            if patient_value is not None:
                if condition(patient_value):
                    print(f"Contraindication Alert: {medication_name} is contraindicated due to {param} level of {patient_value}.")
                    return False
            else:
                print(f"Warning: {param} value not available for Patient {patient['PatientID']}.")
    return True

def monitor_lab_values(patient):
    alerts = []
    for med in patient['Medications']:
        if med in medications:
            med_info = medications[med]
            monitoring_params = med_info.get('MonitoringParameters', [])
            for param in monitoring_params:
                patient_value = patient['LabResults'].get(param)
                if patient_value is not None:
                    # Check upper and lower limits
                    upper_limit = upper_limits_normal.get(param)
                    lower_limit = lower_limits_normal.get(param)
                    if upper_limit and patient_value > upper_limit:
                        alerts.append(f"{param} is above normal range at {patient_value}.")
                    if lower_limit and patient_value < lower_limit:
                        alerts.append(f"{param} is below normal range at {patient_value}.")
                else:
                    alerts.append(f"{param} value not available for monitoring.")
    return alerts

def find_patient(patient_id):
    for patient in patients:
        if patient['PatientID'] == patient_id:
            return patient
    return None

def prescribe_medication(patient, medication_name):
    if not check_contraindications(patient, medication_name):
        print(f"Cannot prescribe '{medication_name}' to Patient {patient['PatientID']}.")
        return

    if medication_name in patient['Medications']:
        print(f"Patient {patient['PatientID']} is already taking '{medication_name}'.")
        return

    patient['Medications'].append(medication_name)
    print(f"Medication '{medication_name}' prescribed to Patient {patient['PatientID']}.")

    # Display side effects
    side_effects = medications[medication_name].get('SideEffects', [])
    if side_effects:
        print(f"Common side effects of {medication_name}: {', '.join(side_effects)}")

def display_patient_info(patient):
    print(f"Patient ID: {patient['PatientID']}")
    print(f"Name: {patient['FirstName']} {patient['LastName']}")
    print(f"Age: {patient['Age']}")
    print(f"Gender: {patient['Gender']}")
    allergies = ', '.join(patient.get('Allergies', [])) if patient.get('Allergies') else 'None'
    print(f"Allergies: {allergies}")
    medical_history = ', '.join(patient.get('MedicalHistory', [])) if patient.get('MedicalHistory') else 'None'
    print(f"Medical History: {medical_history}")
    print(f"Lab Results:")
    for lab, value in patient['LabResults'].items():
        print(f"  {lab}: {value}")
    medications_list = ', '.join(patient['Medications']) if patient['Medications'] else 'None'
    print(f"Current Medications: {medications_list}")

def main():
    while True:
        print("""
Clinical Decision Support System
1. View Patient Information
2. Prescribe Medication
3. Monitor Patient Lab Values
4. Exit
Enter choice:""")
        choice = input().strip()

        if choice == '1':
            try:
                patient_id = int(input("Enter Patient ID: ").strip())
            except ValueError:
                print("Invalid Patient ID. Please enter a number.")
                continue
            patient = find_patient(patient_id)
            if patient:
                display_patient_info(patient)
            else:
                print(f"Patient with ID {patient_id} not found.")
        elif choice == '2':
            try:
                patient_id = int(input("Enter Patient ID: ").strip())
            except ValueError:
                print("Invalid Patient ID. Please enter a number.")
                continue
            patient = find_patient(patient_id)
            if patient:
                medication_name = input("Enter Medication Name: ").strip()
                prescribe_medication(patient, medication_name)
            else:
                print(f"Patient with ID {patient_id} not found.")
        elif choice == '3':
            try:
                patient_id = int(input("Enter Patient ID: ").strip())
            except ValueError:
                print("Invalid Patient ID. Please enter a number.")
                continue
            patient = find_patient(patient_id)
            if patient:
                alerts = monitor_lab_values(patient)
                if alerts:
                    print("Lab Monitoring Alerts:")
                    for alert in alerts:
                        print(f"- {alert}")
                else:
                    print("No lab monitoring alerts for this patient.")
            else:
                print(f"Patient with ID {patient_id} not found.")
        elif choice == '4':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
