import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk, __version__ as PILLOW_VERSION
from patients import patients
from medications import medications
import time
import threading
import re
import difflib
import os
import sys

# Tooltip Class
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            tw, text=self.text, justify='left',
            background="#ffffe0", relief='solid', borderwidth=1,
            font=("tahoma", "8", "normal")
        )
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


# GUI Application Class
class CDSSApp:
    def __init__(self, root, base_dir):
        self.root = root
        self.root.title("AI Clinical Decision Support System")
        self.root.geometry("800x700")
        self.root.configure(bg='white')
        self.patients = patients
        self.selected_patient = None
        self.base_dir = base_dir

        self.images = {}

        if int(PILLOW_VERSION.split('.')[0]) >= 10:
            self.resample_filter = Image.Resampling.LANCZOS
        else:
            self.resample_filter = Image.ANTIALIAS

        self.create_widgets()
        self.load_images()

    def load_images(self):
        try:
            footer_image_path = os.path.join(self.base_dir, 'images', 'download.jpg')

            print(f"Attempting to load footer image from: {footer_image_path}")

            if not os.path.isfile(footer_image_path):
                raise FileNotFoundError(f"Footer image not found at {footer_image_path}")

            img = Image.open(footer_image_path)
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            resized_img = img.resize((60, 60), resample=self.resample_filter)
            self.images['footer_image'] = ImageTk.PhotoImage(resized_img)

            print("Footer image loaded successfully.")

        except Exception as e:
            messagebox.showerror("Image Loading Error", f"An error occurred while loading images:\n{e}")
            sys.exit(1)

    def create_widgets(self):
        size_label = ttk.Label(self.root, text="Window Size: 800x700", background='white')
        size_label.pack(pady=(10, 0))

        patient_frame = ttk.Frame(self.root)
        patient_frame.pack(pady=10)

        ttk.Label(patient_frame, text="Select Patient:").pack(side=tk.LEFT)

        patient_ids = [patient['PatientID'] for patient in self.patients]
        self.patient_var = tk.IntVar()
        self.patient_dropdown = ttk.Combobox(patient_frame, textvariable=self.patient_var, state="readonly")
        self.patient_dropdown['values'] = patient_ids
        self.patient_dropdown.bind("<<ComboboxSelected>>", self.load_patient_info)
        self.patient_dropdown.pack(side=tk.LEFT, padx=5)

        self.info_text = tk.Text(self.root, height=10, width=80, wrap='word', bg='lightyellow')
        self.info_text.pack(pady=10)

        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(pady=10)

        update_lab_button = ttk.Button(
            buttons_frame,
            text="Update Lab Results",
            command=self.update_labs
        )
        update_lab_button.pack(side=tk.LEFT, padx=5)

        prescribe_med_button = ttk.Button(
            buttons_frame,
            text="Prescribe Medication",
            command=self.prescribe_medication
        )
        prescribe_med_button.pack(side=tk.LEFT, padx=5)

        open_chatbot_button = ttk.Button(buttons_frame, text="Open AI Assistant", command=self.open_chatbot)
        open_chatbot_button.pack(side=tk.LEFT, padx=5)

        self.warnings_text = tk.Text(self.root, height=6, width=80, fg='red', wrap='word', bg='lightpink')
        self.warnings_text.pack(pady=10)

        self.recommendations_text = tk.Text(self.root, height=6, width=80, fg='blue', wrap='word', bg='lightblue')
        self.recommendations_text.pack(pady=10)

        self.risk_frame = ttk.Frame(self.root)
        self.risk_frame.pack(pady=10)

        footer_label = ttk.Label(self.root, image=self.images.get('footer_image'), borderwidth=2, relief="solid")
        footer_label.image = self.images.get('footer_image')
        footer_label.pack(side=tk.BOTTOM, anchor='e', padx=10, pady=10)

        print("Footer image placed at bottom right using pack.")

    def load_patient_info(self, event=None):
        patient_id = self.patient_var.get()
        self.selected_patient = find_patient(patient_id)
        if self.selected_patient:
            self.display_patient_info()
            self.display_warnings()
            self.display_recommendations()
            self.display_clinical_risk_scores()

    def display_patient_info(self):
        patient = self.selected_patient
        self.info_text.delete(1.0, tk.END)
        info = f"Patient ID: {patient['PatientID']}\n"
        info += f"Name: {patient['FirstName']} {patient['LastName']}\n"
        info += f"Age: {patient['Age']}\n"
        info += f"Gender: {patient['Gender']}\n"
        info += f"Lab Results:\n"
        for lab, value in patient['LabResults'].items():
            info += f"  {lab}: {value}\n"
        medications_list = ', '.join(patient['Medications']) if patient['Medications'] else 'None'
        info += f"Current Medications: {medications_list}\n"
        self.info_text.insert(tk.END, info)

    def display_warnings(self):
        warnings = check_lab_warnings(self.selected_patient)
        self.warnings_text.delete(1.0, tk.END)
        if warnings:
            message = "Please check the following prescriptions:\n\n" + "\n".join(warnings)
            self.warnings_text.insert(tk.END, message)
        else:
            self.warnings_text.insert(tk.END, "No warnings.")

    def display_recommendations(self):
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, "Generating recommendations...\n")

        progress = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='indeterminate')
        progress.pack(pady=5)
        progress.start(10)

        def generate_and_display():
            recommendations = generate_recommendations(self.selected_patient)
            progress.stop()
            progress.pack_forget()
            self.recommendations_text.delete(1.0, tk.END)
            if recommendations:
                message = "AI Recommendations:\n\n" + "\n".join(recommendations)
                self.recommendations_text.insert(tk.END, message)
            else:
                self.recommendations_text.insert(tk.END, "No recommendations at this time.")

        threading.Thread(target=generate_and_display).start()

    def display_clinical_risk_scores(self):
        # Clear previous risk scores
        for widget in self.risk_frame.winfo_children():
            widget.destroy()

        # Calculate and display cardiovascular risk score
        cv_risk = calculate_cardiovascular_risk_score(self.selected_patient)
        if cv_risk['is_elevated']:
            cv_label = tk.Label(self.risk_frame, text=f"Cardiovascular Risk: {cv_risk['score']}%", fg='red')
            cv_label.pack(side=tk.LEFT, padx=10)

        # Calculate and display renal risk score
        renal_risk = calculate_renal_risk_score(self.selected_patient)
        if renal_risk['is_elevated']:
            renal_label = tk.Label(self.risk_frame, text=f"Renal Risk: {renal_risk['score']}%", fg='red')
            renal_label.pack(side=tk.LEFT, padx=10)

        # Calculate and display lung risk score
        lung_risk = calculate_lung_risk_score(self.selected_patient)
        if lung_risk['is_elevated']:
            lung_label = tk.Label(self.risk_frame, text=f"Lung Risk: {lung_risk['score']}%", fg='red')
            lung_label.pack(side=tk.LEFT, padx=10)

        # Calculate and display liver risk score
        liver_risk = calculate_liver_risk_score(self.selected_patient)
        if liver_risk['is_elevated']:
            liver_label = tk.Label(self.risk_frame, text=f"Liver Risk: {liver_risk['score']}%", fg='red')
            liver_label.pack(side=tk.LEFT, padx=10)

    def update_labs(self):
        if not self.selected_patient:
            messagebox.showerror("Error", "Please select a patient first.")
            return

        lab_window = tk.Toplevel(self.root)
        lab_window.title("Update Lab Results")

        lab_entries = {}
        row = 0
        for lab in self.selected_patient['LabResults'].keys():
            ttk.Label(lab_window, text=lab).grid(row=row, column=0, padx=10, pady=5, sticky='e')
            entry = ttk.Entry(lab_window)
            entry.insert(0, str(self.selected_patient['LabResults'][lab]))
            entry.grid(row=row, column=1, padx=10, pady=5)
            lab_entries[lab] = entry
            row += 1

        def save_labs():
            new_lab_results = {}
            for lab, entry in lab_entries.items():
                try:
                    value = float(entry.get())
                    new_lab_results[lab] = value
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {lab}. Please enter a numeric value.")
                    return
            update_patient_labs(self.selected_patient, new_lab_results)
            self.display_patient_info()
            self.display_warnings()
            self.display_recommendations()
            self.display_clinical_risk_scores()
            lab_window.destroy()

        ttk.Button(lab_window, text="Save", command=save_labs).grid(row=row, column=0, columnspan=2, pady=10)

    def prescribe_medication(self):
        if not self.selected_patient:
            messagebox.showerror("Error", "Please select a patient first.")
            return

        med_window = tk.Toplevel(self.root)
        med_window.title("Prescribe Medication")

        ttk.Label(med_window, text="Select Medication:").pack(pady=5)
        med_name_var = tk.StringVar()
        medications_list = sorted(medications.keys())
        med_dropdown = ttk.Combobox(med_window, textvariable=med_name_var, values=medications_list, state="readonly")
        med_dropdown.pack(pady=5)

        def prescribe():
            med_name = med_name_var.get()
            if med_name:
                result = prescribe_medication_gui(self.selected_patient, med_name)
                if result:
                    self.display_patient_info()
                    self.display_warnings()
                    self.display_recommendations()
                    self.display_clinical_risk_scores()
                med_window.destroy()
            else:
                messagebox.showerror("Error", "Please select a medication.")

        ttk.Button(med_window, text="Prescribe", command=prescribe).pack(pady=10)

    def open_chatbot(self):
        chatbot_window = tk.Toplevel(self.root)
        chatbot_window.title("AI Assistant")
        chatbot_window.geometry("400x500")
        chatbot_window.configure(bg='white')

        chat_display = tk.Text(chatbot_window, height=20, width=50, wrap='word', bg='lightyellow')
        chat_display.pack(pady=5, padx=5)

        input_frame = ttk.Frame(chatbot_window)
        input_frame.pack(pady=10, padx=5, fill=tk.X)

        user_input_var = tk.StringVar()
        user_input_entry = ttk.Entry(input_frame, textvariable=user_input_var, width=40)
        user_input_entry.pack(side=tk.LEFT, padx=(0, 5), pady=5, expand=True, fill=tk.X)

        send_button = ttk.Button(input_frame, text="Send", command=lambda: self.send_message(chat_display, user_input_var))
        send_button.pack(side=tk.LEFT, padx=(0, 5), pady=5)

        if 'footer_image' in self.images:
            footer_image_label = ttk.Label(input_frame, image=self.images['footer_image'])
            footer_image_label.image = self.images['footer_image']
            footer_image_label.pack(side=tk.LEFT, padx=(5, 0), pady=5)
        else:
            print("Footer image not found in self.images.")

        chatbot_window.bind('<Return>', lambda event: self.send_message(chat_display, user_input_var))

        user_input_entry.focus()

    def send_message(self, chat_display, user_input_var):
        user_message = user_input_var.get().strip()
        if user_message:
            chat_display.insert(tk.END, f"You: {user_message}\n")
            response = self.chatbot_response(user_message)
            chat_display.insert(tk.END, f"AI Assistant: {response}\n")
            chat_display.see(tk.END)
            user_input_var.set('')

    def chatbot_response(self, message):
        message = message.lower()

        if re.search(r'\b(hello|hi|hey)\b', message):
            return "Hello! How can I assist you today?"

        elif "patient condition" in message:
            return "The patient's condition is stable."

        elif "recommendation" in message:
            recommendations = generate_recommendations(self.selected_patient)
            if recommendations:
                return "Based on my analysis, " + "; ".join(recommendations)
            else:
                return "No specific recommendations at this time."

        elif "risk assessment" in message or "clinical risk score" in message:
            risk_message = calculate_overall_risk_score(self.selected_patient)
            return risk_message

        elif "side effect" in message or "side effects" in message:
            meds_in_message = self.extract_medications(message)
            if meds_in_message:
                responses = []
                for med in meds_in_message:
                    side_effects = medications[med].get('SideEffects', [])
                    if side_effects:
                        response = f"Common side effects of {med} include: " + ", ".join(side_effects) + "."
                        responses.append(response)
                    else:
                        responses.append(f"Sorry, I don't have side effect information for {med}.")
                return " ".join(responses)
            else:
                return "Please specify the medication you are inquiring about."

        elif "information about" in message or "tell me about" in message or "what should i look out for" in message:
            meds_in_message = self.extract_medications(message)
            if meds_in_message:
                responses = []
                for med in meds_in_message:
                    info = self.get_medication_info(med)
                    responses.append(info)
                return "\n".join(responses)
            else:
                return "Please specify the medication you are interested in."

        elif "thank you" in message or "thanks" in message:
            return "You're welcome! Let me know if you need anything else."

        else:
            return "I'm sorry, I didn't understand that. You can ask me about medication side effects or for recommendations."

    def extract_medications(self, message):
        meds_in_message = []
        words = re.findall(r'\b\w+\b', message.lower())
        for word in words:
            matches = difflib.get_close_matches(word, [med.lower() for med in medications.keys()], cutoff=0.8)
            for match in matches:
                med_name = next((med for med in medications.keys() if med.lower() == match), None)
                if med_name and med_name not in meds_in_message:
                    meds_in_message.append(med_name)
        return meds_in_message

    def get_medication_info(self, med_name):
        if med_name in medications:
            med_info = medications[med_name]
            side_effects = med_info.get('SideEffects', [])
            monitoring_params = med_info.get('MonitoringParameters', {})
            info = f"{med_name}:\n"
            if side_effects:
                info += "Common side effects include: " + ", ".join(side_effects) + ".\n"
            if monitoring_params:
                params = ', '.join(monitoring_params.keys())
                info += "Important parameters to monitor: " + params + ".\n"

            med_name_formatted = med_name.lower().replace(' ', '')
            info += f"More information: https://medlineplus.gov/druginfo/meds/a{med_name_formatted}.html\n"
            return info
        else:
            return f"Sorry, I don't have information about {med_name}."


# Function to check and warn about lab values exceeding safety limits
def check_lab_warnings(patient):
    warnings = []
    for medication in patient['Medications']:
        if medication in medications:
            med_info = medications[medication]
            monitoring_params = med_info.get('MonitoringParameters', {})
            contraindications = med_info.get('Contraindications', {})
            for lab_name, limits in monitoring_params.items():
                patient_value = patient['LabResults'].get(lab_name)
                if patient_value is not None:
                    max_value = limits.get('max')
                    min_value = limits.get('min')
                    if (max_value is not None and patient_value > max_value) or (min_value is not None and patient_value < min_value):
                        warnings.append(f"{medication} ({lab_name} is {patient_value}, which may pose a safety risk)")
            for param, condition in contraindications.items():
                if param in patient['LabResults']:
                    patient_value = patient['LabResults'].get(param)
                elif param in patient.get('Conditions', {}):
                    patient_value = patient['Conditions'].get(param)
                else:
                    patient_value = None

                if patient_value is not None and condition(patient_value):
                    warnings.append(f"{medication} is contraindicated due to {param}.")
    return warnings


# Function to update patient lab results
def update_patient_labs(patient, new_lab_results):
    patient['LabResults'].update(new_lab_results)
    warnings = check_lab_warnings(patient)
    if warnings:
        message = "Please check the following prescriptions:\n\n" + "\n".join(warnings)
        messagebox.showwarning("Medications at Risk", message)


# Function to find a patient by ID
def find_patient(patient_id):
    for patient in patients:
        if patient['PatientID'] == patient_id:
            return patient
    return None


# Function to prescribe medication with checks for contraindications
def prescribe_medication_gui(patient, medication_name):
    if medication_name not in medications:
        messagebox.showerror("Error", f"Medication '{medication_name}' not found.")
        return False
    med_info = medications[medication_name]
    contraindications = med_info.get('Contraindications', {})
    for param, condition in contraindications.items():
        if param in patient['LabResults']:
            patient_value = patient['LabResults'].get(param)
        elif param in patient.get('Conditions', {}):
            patient_value = patient['Conditions'].get(param)
        else:
            patient_value = None

        if patient_value is not None and condition(patient_value):
            messagebox.showwarning("Contraindication Alert", f"{medication_name} is contraindicated due to {param}.")
            return False
    if medication_name not in patient['Medications']:
        patient['Medications'].append(medication_name)
        messagebox.showinfo("Medication Prescribed", f"{medication_name} has been prescribed to the patient.")
        return True
    else:
        messagebox.showinfo("Already Prescribed", f"{medication_name} is already prescribed to the patient.")
        return False


# Function to generate recommendations based on patient data
def generate_recommendations(patient):
    recommendations = []
    time.sleep(1)

    blood_pressure = patient['LabResults'].get('Blood Pressure')
    if blood_pressure and blood_pressure > 140:
        recommendations.append("Consider adjusting antihypertensive medications.")

    hba1c = patient['LabResults'].get('Hemoglobin A1C')
    if hba1c and hba1c > 7.0:
        recommendations.append("Recommend optimizing glycemic control.")

    ldl = patient['LabResults'].get('Cholesterol (LDL)')
    if ldl and ldl > 100:
        recommendations.append("Suggest increasing statin therapy.")

    potassium = patient['LabResults'].get('Potassium')
    if potassium and potassium > 5.5:
        recommendations.append("Monitor for hyperkalemia; adjust medications accordingly.")

    inr = patient['LabResults'].get('INR')
    if inr and inr > 3.0 and 'Warfarin' in patient['Medications']:
        recommendations.append("INR is high; consider adjusting Warfarin dosage.")

    return recommendations


# Function to calculate cardiovascular risk score
def calculate_cardiovascular_risk_score(patient):
    risk_factors = 0
    age = patient['Age']
    if age > 50:
        risk_factors += 1
    blood_pressure = patient['LabResults'].get('Blood Pressure', 0)
    if blood_pressure > 140:
        risk_factors += 1
    ldl = patient['LabResults'].get('Cholesterol (LDL)', 0)
    if ldl > 130:
        risk_factors += 1
    smoking_status = patient.get('SmokingStatus', False)
    if smoking_status:
        risk_factors += 1

    risk_percentage = (risk_factors / 4) * 100
    return {'score': f"{risk_percentage:.0f}", 'is_elevated': risk_percentage > 25}


# Function to calculate renal risk score
def calculate_renal_risk_score(patient):
    risk_factors = 0
    creatinine = patient['LabResults'].get('Creatinine', 0)
    if creatinine > 1.5:
        risk_factors += 1
    gfr = patient['LabResults'].get('GFR', 0)
    if gfr < 60:
        risk_factors += 1

    risk_percentage = (risk_factors / 2) * 100
    return {'score': f"{risk_percentage:.0f}", 'is_elevated': risk_percentage > 25}


# Function to calculate lung risk score
def calculate_lung_risk_score(patient):
    risk_factors = 0
    fev1_fvc = patient['LabResults'].get('FEV1/FVC Ratio', 0)
    if fev1_fvc < 0.7:
        risk_factors += 1
    history_copd = patient.get('HistoryOfCOPD', False)
    if history_copd:
        risk_factors += 1

    risk_percentage = (risk_factors / 2) * 100
    return {'score': f"{risk_percentage:.0f}", 'is_elevated': risk_percentage > 25}

# Function to calculate liver risk score
def calculate_liver_risk_score(patient):
    risk_factors = 0
    liver_enzymes = patient['LabResults'].get('Liver Enzymes', 0)
    if liver_enzymes > 70:
        risk_factors += 1
    bilirubin = patient['LabResults'].get('Bilirubin', 0)
    if bilirubin > 1.2:
        risk_factors += 1

    risk_percentage = (risk_factors / 2) * 100
    return {'score': f"{risk_percentage:.0f}", 'is_elevated': risk_percentage > 25}



def display_clinical_risk_scores(self):
    # Clear previous risk scores
    for widget in self.risk_frame.winfo_children():
        widget.destroy()

    # Calculate and display cardiovascular risk score
    cv_risk = calculate_cardiovascular_risk_score(self.selected_patient)
    if cv_risk['is_elevated']:
        cv_label = tk.Label(self.risk_frame, text=f"Cardiovascular Risk: {cv_risk['score']}%", fg='red')
        cv_label.pack(side=tk.LEFT, padx=10)
        # Add tooltip for cardiovascular risk score
        CreateToolTip(cv_label, "Cardiovascular Risk is calculated based on age, blood pressure, cholesterol (LDL), and smoking status.")

    # Calculate and display renal risk score
    renal_risk = calculate_renal_risk_score(self.selected_patient)
    if renal_risk['is_elevated']:
        renal_label = tk.Label(self.risk_frame, text=f"Renal Risk: {renal_risk['score']}%", fg='red')
        renal_label.pack(side=tk.LEFT, padx=10)
        # Add tooltip for renal risk score
        CreateToolTip(renal_label, "Renal Risk is calculated based on creatinine and GFR levels.")

    # Calculate and display lung risk score
    lung_risk = calculate_lung_risk_score(self.selected_patient)
    if lung_risk['is_elevated']:
        lung_label = tk.Label(self.risk_frame, text=f"Lung Risk: {lung_risk['score']}%", fg='red')
        lung_label.pack(side=tk.LEFT, padx=10)
        # Add tooltip for lung risk score
        CreateToolTip(lung_label, "Lung Risk is calculated based on FEV1/FVC ratio and history of COPD.")

    # Calculate and display liver risk score
    liver_risk = calculate_liver_risk_score(self.selected_patient)
    if liver_risk['is_elevated']:
        liver_label = tk.Label(self.risk_frame, text=f"Liver Risk: {liver_risk['score']}%", fg='red')
        liver_label.pack(side=tk.LEFT, padx=10)
        # Add tooltip for liver risk score
        CreateToolTip(liver_label, "Liver Risk is calculated based on liver enzyme and bilirubin levels.")

def create_widgets(self):
    # Add a basic label to test tooltips
    test_label = tk.Label(self.root, text="Hover over me for a tooltip!", fg='blue')
    test_label.pack(pady=10)
    CreateToolTip(test_label, "This is a test tooltip for the label.")
    
    # Other widget creation code...




if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    root = tk.Tk()

    images_folder = os.path.join(base_dir, 'images')
    print(f"Looking for images folder at: {images_folder}")

    if not os.path.exists(images_folder):
        messagebox.showerror("Missing Folder", f"The 'images' folder is missing at {images_folder}. Please create it and add the required images.")
        sys.exit(1)
    else:
        print("Images folder found.")
        app = CDSSApp(root, base_dir)
        root.mainloop()
