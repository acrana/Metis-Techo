# Healthcare Technology Portfolio

Welcome to my portfolio exploring the intersection of clinical informatics, AI safety, and health data interoperability. These projects focus on solving real-world healthcare friction points—bridging clinical workflow constraints with robust engineering standards in Python, SQL, PyTorch, and HL7 FHIR.

---

## Featured Core Platforms

### [METIS / AICDSS (Clinical Decision Support System)](https://github.com/acrana/Metis-Techo/tree/main/AICDSS)
An intelligent clinical decision support application designed to assist healthcare providers with real-time risk assessment and medication safety protocols. 
- **Core Stack:** Python, PySide6, SQLite, PyTorch, Pandas
- **Key Capabilities:**
  - Real-time patient vital sign and lab monitoring with customizable risk alerts.
  - Medication safety engine checking for drug interactions and clinical contraindications.
  - Interactive dark-mode desktop UI designed for low-friction clinician interaction.

### [CLABSI Risk Prediction Pipeline](https://github.com/acrana/Metis-Techo/tree/main/CLABSI)
An explainable machine learning pipeline utilizing the MIMIC-IV critical care database to predict Central Line-Associated Bloodstream Infection (CLABSI) risk.
- **Core Stack:** Python, Scikit-Learn, XGBoost, MIMIC-IV (PostgreSQL)
- **Key Capabilities:**
  - Rigorous data cleaning, cohort matching, and outlier management on complex critical care datasets.
  - Optimized XGBoost classifier achieving strong ROC-AUC performance for early clinical intervention.
  - **In Progress:** Integrating HL7 FHIR R4 API endpoints for real-time risk scoring.

---

## Advanced Specialty Projects (In Development)

### [Themis-Kansa](https://github.com/acrana/Themis-Kansa) *(Clinical AI Governance & Audit)*
- **Focus:** AI Model Validation & Fairness Auditing
- **Description:** Evaluates ICU scoring and mortality models (such as SAPS II) on MIMIC-IV data to simulate a hospital AI governance committee audit. Tests for demographic parity (race, sex, insurance, care unit), calibration (Brier scores), discrimination (ROC-AUC), and temporal drift, resulting in a formal Model Card and deployment recommendation.

### [Panacea-Shogo](https://github.com/acrana/Panacea-Shogo) *(EHR Interoperability)*
- **Focus:** SMART on FHIR Application & Medication Reconciliation
- **Description:** A secure web-launchable SMART on FHIR application utilizing OAuth2 authentication and the RxNav API. Ingests patient home medications and inpatient orders from a FHIR sandbox, normalizes nomenclature to RxNorm, and flags omissions, duplications, and dose discrepancies.

### [Hermes-Kakehashi](https://github.com/acrana/Hermes-Kakehashi) *(Cross-Border Terminology Bridge)*
- **Focus:** International Health Data & Regulatory Mapping
- **Description:** A specialized translation and compliance tool mapping US RxNorm active ingredients to Japanese pharmaceutical nomenclature (YJ codes and generic brand names), while automatically auditing traveler import restrictions and controlled substance/stimulant bans.

---

## Technical Stack & Competencies
- **Clinical Standards:** HL7 FHIR R4, RxNorm, LOINC, ICD-10, OAuth2
- **Languages & Frameworks:** Python (Pandas, NumPy, Scikit-Learn, PyTorch, FastAPI, PySide6), SQL (SQLite, PostgreSQL)
- **Clinical Domains:** Critical Care Informatics, CDS Governance, EHR Workflow Optimization

---
