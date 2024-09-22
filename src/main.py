# src/main.py

from data_access import (
    get_patient_demographics,
    get_patient_surveys,
    get_patient_ade_records,
    get_medication_info,
    add_prescription
)
from risk_assessment import get_patient_risk_assessment

def main():
    while True:
        print("""
Clinical Decision Support System
1. View Patient Demographics
2. View Patient Surveys
3. View Patient ADE Records
4. Assess Patient Risk
5. Add Medication and Assess Risk
6. Exit
Enter choice:""")
        choice = input().strip()

        if choice == '1':
            patient_id = input("Enter Patient ID: ").strip()
            if not patient_id.isdigit():
                print("Invalid Patient ID. Please enter a numerical value.")
                continue
            demographics = get_patient_demographics(patient_id)
            if demographics.empty:
                print(f"No demographics found for Patient ID {patient_id}.")
            else:
                print(demographics)

        elif choice == '2':
            patient_id = input("Enter Patient ID: ").strip()
            if not patient_id.isdigit():
                print("Invalid Patient ID. Please enter a numerical value.")
                continue
            surveys = get_patient_surveys(patient_id)
            if surveys.empty:
                print(f"No surveys found for Patient ID {patient_id}.")
            else:
                print(surveys)

        elif choice == '3':
            patient_id = input("Enter Patient ID: ").strip()
            if not patient_id.isdigit():
                print("Invalid Patient ID. Please enter a numerical value.")
                continue
            ade_records = get_patient_ade_records(patient_id)
            if ade_records.empty:
                print(f"No ADE records found for Patient ID {patient_id}.")
            else:
                print(ade_records)

        elif choice == '4':
            patient_id = input("Enter Patient ID: ").strip()
            if not patient_id.isdigit():
                print("Invalid Patient ID. Please enter a numerical value.")
                continue
            assessment = get_patient_risk_assessment(int(patient_id), 'None')  # 'None' indicates no new medication
            if assessment is None:
                print("Risk assessment could not be performed.")
            else:
                print(f"""
Patient ID: {assessment['PatientID']}
Predicted ADE Risk: {assessment['RiskLevel']}
Risk Probability: {assessment['RiskProbability']:.2f}
""")

        elif choice == '5':
            patient_id = input("Enter Patient ID: ").strip()
            if not patient_id.isdigit():
                print("Invalid Patient ID. Please enter a numerical value.")
                continue
            medication_name = input("Enter Medication Name: ").strip()

            # Check if medication exists
            medication_info = get_medication_info(medication_name)
            if medication_info.empty:
                print(f"Medication '{medication_name}' not found in database.")
                continue

            # Optional: Check for known risk factors before adding prescription
            risk_factors = medication_info['RiskFactors'].iloc[0].split(', ')
            patient_demographics = get_patient_demographics(patient_id)
            if patient_demographics.empty:
                print(f"No demographics found for Patient ID {patient_id}.")
                continue

            # Example: Simple risk factor check (can be expanded)
            # Here, we assume 'Pregnancy' is a critical risk factor
            # Since we don't have pregnancy data, this is a placeholder
            if 'Pregnancy' in risk_factors:
                print(f"Warning: {medication_name} is contraindicated in pregnancy.")
                # Decide whether to proceed or not
                proceed = input("Do you still want to add this medication? (y/n): ").strip().lower()
                if proceed != 'y':
                    print("Medication not added.")
                    continue

            # Add prescription
            added = add_prescription(int(patient_id), medication_name)
            if not added:
                continue

            # Assess risk with the new medication
            assessment = get_patient_risk_assessment(int(patient_id), medication_name)
            if assessment is None:
                print("Risk assessment could not be performed.")
            else:
                print(f"""
Patient ID: {assessment['PatientID']}
Medication: {medication_name}
Predicted ADE Risk: {assessment['RiskLevel']}
Risk Probability: {assessment['RiskProbability']:.2f}
""")

        elif choice == '6':
            print("Exiting the application.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
