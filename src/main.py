from patients import patients
from medications import medications, upper_limits_normal, lower_limits_normal
from risk_assessment import get_patient_risk_assessment  # Import the risk assessment function

def prescribe_medication(patient, medication_name):
    # Check if the medication exists in the database
    if medication_name not in medications:
        print(f"Medication '{medication_name}' not found.")
        return

    # Perform risk assessment
    risk_info = get_patient_risk_assessment(patient['PatientID'], medication_name)
    if risk_info is None:
        print(f"Risk assessment failed for Patient {patient['PatientID']} with medication '{medication_name}'.")
        return

    # Display the risk level and probability
    print(f"Risk Assessment for Patient {patient['PatientID']} on '{medication_name}':")
    print(f"  Risk Level: {risk_info['RiskLevel']}")
    print(f"  Risk Probability: {risk_info['RiskProbability']:.2f}")

    # Prevent the prescription if the risk is too high
    if risk_info['RiskLevel'] == 'High':
        print(f"Warning: Prescription of '{medication_name}' for Patient {patient['PatientID']} is not allowed due to high risk.")
        return
    elif risk_info['RiskLevel'] == 'Moderate':
        print(f"Caution: '{medication_name}' for Patient {patient['PatientID']} is associated with moderate risk.")

    # If risk is low or acceptable, proceed with prescription
    if medication_name in patient['Medications']:
        print(f"Patient {patient['PatientID']} is already taking '{medication_name}'.")
        return

    # Add medication to the patient's list
    patient['Medications'].append(medication_name)
    print(f"Medication '{medication_name}' prescribed to Patient {patient['PatientID']}.")

    # Display side effects of the medication
    side_effects = medications[medication_name].get('SideEffects', [])
    if side_effects:
        print(f"Common side effects of {medication_name}: {', '.join(side_effects)}")

