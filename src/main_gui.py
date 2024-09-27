import tkinter as tk
from tkinter import messagebox
from patients import patients
from medications import medications
from main import prescribe_and_monitor_gui, update_patient_labs, find_patient

# Create the main application window
root = tk.Tk()
root.title("AI CDSS - Clinical Decision Support System")

# Global variable for the current patient
current_patient = None

# Create a selectable dropdown list for medications
medication_options = list(medications.keys())  # Get all medication names as options
selected_medication = tk.StringVar(root)
selected_medication.set(medication_options[0])  # Set the default option

# Create a dropdown (OptionMenu) for selecting a medication
dropdown_medications = tk.OptionMenu(root, selected_medication, *medication_options)
dropdown_medications.grid(row=2, column=1, padx=10, pady=10)

# Function to display patient information
def view_patient_info():
    patient_id = entry_patient_id.get()
    if patient_id.isdigit():
        global current_patient
        current_patient = find_patient(int(patient_id))
        if current_patient:
            info = f"Patient ID: {current_patient['PatientID']}\n"
            info += f"Name: {current_patient['FirstName']} {current_patient['LastName']}\n"
            info += f"Age: {current_patient['Age']}\n"
            info += f"Gender: {current_patient['Gender']}\n"
            info += f"Lab Results: {current_patient['LabResults']}\n"
            info += f"Current Medications: {', '.join(current_patient['Medications'])}\n"
            text_patient_info.config(state=tk.NORMAL)
            text_patient_info.delete(1.0, tk.END)  # Clear existing info
            text_patient_info.insert(tk.END, info)  # Display new info
            text_patient_info.config(state=tk.DISABLED)
        else:
            messagebox.showerror("Error", "Patient not found!")
    else:
        messagebox.showerror("Error", "Please enter a valid patient ID.")

# Function to prescribe medication
def prescribe_medication():
    if not current_patient:
        messagebox.showerror("Error", "Please select a patient first.")
        return

    medication_name = selected_medication.get()  # Use the selected medication from the dropdown
    
    if medication_name in medications:
        # Run the prescription and monitoring logic
        if prescribe_and_monitor_gui(current_patient, medication_name):
            messagebox.showinfo("Success", f"Medication '{medication_name}' prescribed to Patient {current_patient['PatientID']}.")
            view_patient_info()  # Update patient info after prescription
        else:
            # If the prescription was not successful (due to contraindications)
            messagebox.showwarning("Warning", f"Cannot prescribe '{medication_name}' due to contraindications.")
    else:
        messagebox.showerror("Error", "Medication not found in database.")

# Function to update lab results
def update_labs():
    if not current_patient:
        messagebox.showerror("Error", "Please select a patient first.")
        return

    lab_input = entry_lab_results.get()
    try:
        lab_name, lab_value = lab_input.split('=')
        update_patient_labs(current_patient, {lab_name.strip(): float(lab_value.strip())})
        messagebox.showinfo("Success", "Lab results updated.")
        view_patient_info()  # Update patient info after lab update
    except ValueError:
        messagebox.showerror("Error", "Please enter lab results in 'LabName=Value' format.")

# Create labels and input fields for patient


