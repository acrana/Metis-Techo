from patients import patients
from medications import medications

# Prescribe medication and set up continuous monitoring
def prescribe_and_monitor(patient, medication_name):
    if prescribe_medication(patient, medication_name):
        # Start monitoring the patient's labs after prescribing
        monitor_patient_labs(patient, medication_name)

# Continuous monitoring function
def monitor_patient_labs(patient, medication_name):
    med_info = medications[medication_name]
    monitoring_params = med_info.get('MonitoringParameters', [])
    
    print(f"Starting continuous monitoring for {medication_name}...")

    for param in monitoring_params:
        if param in patient['LabResults']:
            patient_value = patient['LabResults'][param]
            # Check if the patient's value is now beyond safe limits
            if param == 'QTc' and patient_value > 470:
                print(f"Alert: Patient {patient['PatientID']} has a high QTc ({patient_value}) while on {medication_name}.")
            if param == 'Creatinine' and patient_value > 2.0:
                print(f"Alert: Patient {patient['PatientID']} has elevated Creatinine ({patient_value}) while on {medication_name}.")
            # Add more specific checks as per medication requirements
        else:
            print(f"Warning: {param} not available for monitoring in Patient {patient['PatientID']}.")

# Function to simulate lab result updates
def update_patient_labs(patient, new_lab_results):
    # Simulate new lab results (e.g., labs getting worse)
    patient['LabResults'].update(new_lab_results)
    print(f"Updated lab results for Patient {patient['PatientID']}: {new_lab_results}")

    # Re-check the labs for medications they are on
    for med in patient['Medications']:
        monitor_patient_labs(patient, med)

# Function to check contraindications before prescribing medication
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
            else:
                alerts.append(f"{param} is missing in the patient data.")

        if alerts:
            print("Monitoring Warnings: ", ', '.join(alerts))

    return True

# Function to prescribe medication
def prescribe_medication(patient, medication_name):
    if not check_contraindications(patient, medication_name):
        print(f"Cannot prescribe '{medication_name}' to Patient {patient['PatientID']}.")
        return False

    if medication_name in patient['Medications']:
        print(f"Patient {patient['PatientID']} is already taking '{medication_name}'.")
        return False

    patient['Medications'].append(medication_name)
    print(f"Medication '{medication_name}' prescribed to Patient {patient['PatientID']}.")

    # Display side effects
    side_effects = medications[medication_name].get('SideEffects', [])
    if side_effects:
        print(f"Common side effects of {medication_name}: {', '.join(side_effects)}")
    
    return True

# Function to display patient information
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

# Find a patient by ID
def find_patient(patient_id):
    for patient in patients:
        if patient['PatientID'] == patient_id:
            return patient
    return None

# Main function to run the system
def main():
    while True:
        print("""
Clinical Decision Support System
1. View Patient Information
2. Prescribe Medication
3. Update Lab Results
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
                prescribe_and_monitor(patient, medication_name)
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
                print("Enter the updated lab results (e.g., QTc=490):")
                new_lab_results = {}
                while True:
                    lab_input = input("Lab result (or type 'done' to finish): ").strip()
                    if lab_input.lower() == 'done':
                        break
                    try:
                        lab_name, lab_value = lab_input.split('=')
                        new_lab_results[lab_name.strip()] = float(lab_value.strip())
                    except ValueError:
                        print("Invalid input. Please enter in 'LabName=Value' format.")
                        continue
                update_patient_labs(patient, new_lab_results)
            else:
                print(f"Patient with ID {patient_id} not found.")

        elif choice == '4':
            print("Exiting the application.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
