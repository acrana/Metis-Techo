from patients import patients
from medications import medications

def check_contraindications(patient, medication_name):
    if medication_name not in medications:
        print(f"Medication '{medication_name}' not found.")
        return False

    med_info = medications[medication_name]
    contraindications = med_info.get('Contraindications', {})
    
    # Check all contraindications
    for param, condition in contraindications.items():
        if param in patient['LabResults']:
            patient_value = patient['LabResults'][param]
            if condition(patient_value):
                print(f"Contraindication Alert: {medication_name} contraindicated due to {param} level of {patient_value}.")
                return False

    # Check for any monitoring parameters that need warnings
    if 'MonitoringParameters' in med_info:
        alerts = []
        for param in med_info['MonitoringParameters']:
            if param in patient['LabResults']:
                patient_value = patient['LabResults'][param]
                print(f"Monitoring {param}: Current value is {patient_value}.")
                # Check specific parameter ranges if necessary
            else:
                alerts.append(f"{param} is missing in the patient data.")

        if alerts:
            print("Monitoring Warnings: ", ', '.join(alerts))

    return True

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

def main():
    while True:
        print("""
Clinical Decision Support System
1. View Patient Information
2. Prescribe Medication
3. Exit
Enter choice:""")
        choice = input().strip()

        if choice == '1':
            patient_id = int(input("Enter Patient ID: ").strip())
            patient = find_patient(patient_id)
            if patient:
                display_patient_info(patient)
            else:
                print(f"Patient with ID {patient_id} not found.")
        elif choice == '2':
            patient_id = int(input("Enter Patient ID: ").strip())
            patient = find_patient(patient_id)
            if patient:
                medication_name = input("Enter Medication Name: ").strip()
                prescribe_medication(patient, medication_name)
            else:
                print(f"Patient with ID {patient_id} not found.")
        elif choice == '3':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

def find_patient(patient_id):
    for patient in patients:
        if patient['PatientID'] == patient_id:
            return patient
    return None

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

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
