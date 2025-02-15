# METIS - Clinical Decision Support System

METIS is an AI-powered Clinical Decision Support System that helps healthcare providers monitor patient data and predict potential risks. The system analyzes patient vitals, lab results, and medications in real-time to provide early warnings and support clinical decisions.

## Features

* **Patient Monitoring**: Track vital signs and lab results with customizable alerts
* **Medication Safety**: Check drug interactions and contraindications
* **Risk Analysis**: ML-powered prediction of adverse events
* **Interactive Dashboard**: Clean, modern interface with dark mode support
* **Clinical Trends**: Visualize patient data over time

## Technologies Used

* **Frontend**: PySide6
* **Backend**: Python, SQLite
* **Machine Learning**: PyTorch
* **Data Analysis**: Pandas, NumPy

## Technology Stack

* Python 3.9+
* PyTorch
* PySide6
* SQLite



##Current:
PHASE 1: Foundation
REBUILD FOUNDATION (Current Focus)

PostgreSQL/MIMIC IV Integration

Single PostgreSQL connection manager
Replace all synthetic data/SQLite code
Build efficient query system for vitals, labs, meds
Create fast patient search and browsing
Implement proper data caching


Medication System with Real Data

Real medication data from MIMIC prescriptions/emar
Actual drug interactions from data
Medication order simulation (read-only)
Drug risk assessment using real patient outcomes
Keep existing UI but feed it real data


Clinical Data Viewer

Comprehensive patient browser
Lab result visualization
Vital sign trending
Clinical note viewing
Make all components FHIR-ready



FUTURE-PROOFING (Built Into Foundation)

FHIR Preparation

Design data models FHIR-compatible from start
Plan NDJSON support in data structures
Use MIMIC's existing LOINC/RxNorm mappings
Make data retrieval FHIR-transformable
Structure exports for FHIR conversion


ML Model Framework

Create extensible prediction framework
Design for CLABSI/stroke model addition
Build flexible data pipeline
Make components pluggable
Plan result storage/visualization
