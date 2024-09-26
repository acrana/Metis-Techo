import logging
from patients import patients
from medications import medications, upper_limits_normal, lower_limits_normal
from risk_assessment import get_patient_risk_assessment
from data_access import get_patient_demographics, get_patient_surveys, get_patient_ade_records

# Configure the logging system
logging.basicConfig(filename='cdss_audit.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to log prescription-related actions
def log_prescription_action(patient, medication_name, action, risk_info=None):
    log_message = f"PatientID: {patient['PatientID']}, Medication: {medication_name}, Action: {action}"
    if risk_info:
        log_message += f", Risk Level: {risk_info['RiskLevel']}, Risk Probability: {risk_info['RiskProbability']:.2f}"
    logging.info(log_message)

# Check contraindications for the patient and medication
def check_contraindications(patient, medication_name):
    if medication_name not in medications:
        print(f"Medication '{medication_name}' not found.")
        return False

    med_info = medications[medication_name]
    contraindications = med_info.get('Contraindications', {})
    
    for param, condition in contraindications.items():
        patient_value = patient['LabResults'].get(param)
        if patient_value is not None and condition(patient_value):
            print(f"Contraindication Alert: '{medication_name}' cannot be prescribed due to {param} value: {patient_value}.")
            return False
    return True

# Check for interactions with current medications
def check_interactions(patient, medication_name):
    if medication_name not in medications:
        print(f"Medication '{medication_name}' not found.")
        return False

    current_meds = patient.get('Medications', [])
    med_info = medications[medication_name]
    interactions = med_info.get('Interactions', [])
    
    for med in current_meds:
        if med in interactions:
            print(f"Drug Interaction Alert: '{medication_name}' interacts with '{med}'.")
            return False
    return True

# Prescribe medication with risk assessment and logging
def prescribe_medication(patient, medication_name):
    if medication_name not in medications:
        print(f"Medication '{medication_name}' not found.")
        log_prescription_action(patient, medication_name, 'Medication Not Found')
        return

    # Check contraindications first
    if not check_contraindications(patient, medication_name):
        print(f"Cannot prescribe '{medication_name}' due to contraindications.")
        log_prescription_action(patient, medication_name, 'Contraindicated')
        return

    # Check for drug interactions
    if not check_interactions(patient, medication_name):
        print(f"Cannot prescribe '{medication_name}' due to drug interactions.")
        log_prescription_action(patient, medication_name, 'Interaction Blocked')
        return

    # Perform risk assessment
    risk_info = get_patient_risk_assessment(patient['PatientID'], medication_name)
    if risk_info is None:
        print(f"Risk assessment failed for Patient {patient['PatientID']} with medication '{medication_name}'.")
        log_prescription_action(patient, medication_name, 'Risk Assessment Failed')
        return

    # Display risk and log it
    print(f"Risk Assessment for Patient {patient['PatientID']} on '{medication_name}':")
    print(f"  Risk Level: {risk_info['RiskLevel']}")
    print(f"  Risk Probability: {risk_info['RiskProbability']:.2f}")

    if risk_info['RiskLevel'] == 'High':
        print(f"Warning: Prescription of '{medication_name}' for Patient {patient['PatientID']} is not allowed due to high risk.")
        log_prescription_action(patient, medication_name, 'Blocked', risk_info)
        return

    if risk_info['RiskLevel'] == 'Moderate':
        print(f"Caution: '{medication_name}' for Patient {patient['PatientID']} is associated with moderate risk.")
        log_prescription_action(patient, medication_name, 'Warning', risk_info)

    if medication_name in patient['Medications']:
        print(f"Patient {patient['PatientID']} is already taking '{medication_name}'.")
        log_prescription_action(patient, medication_name, 'Already Prescribed')
        return

    # Add medication and log the action
    patient['Medications'].append(medication_name)
    print(f"Medication '{medication_name}' prescribed to Patient {patient['PatientID']}.")
    log_prescription_action(patient, medication_name, 'Prescribed', risk_info)

    # Log side effects
    side_effects = medications[medication_name].get('SideEffects', [])
    if side_effects:
        print(f"Common side effects of {medication_name}: {', '.join(side_effects)}")
        log_prescription_action(patient, medication_name, 'SideEffects: ' + ', '.join(side_effects))

# Find patient by ID
def find_patient(patient_id):
    for patient in patients:
        if patient['PatientID'] == patient_id:
            return patient
    return None

# Display patient information
def display_patient_info(patient):
    print(f"Patient ID: {patient['PatientID']}")
    print(f"Name: {patient['FirstName']} {patient['LastName']}")
    print(f"Age: {patient['Age']}")
    print(f"Gender: {patient['Gender']}")
    print(f"Allergies: {', '.join(patient.get('Allergies', [])) or 'None'}")
    print(f"Medical History: {', '.join(patient.get('MedicalHistory', [])) or 'None'}")
    print("Lab Results:")
    for lab, value in patient['LabResults'].items():
        print(f"  {lab}: {value}")
    print(f"Medications: {', '.join(patient['Medications']) or 'None'}")

# Main function with a simple command-line interface
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
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
