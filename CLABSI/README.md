``` mermaid
erDiagram
    patients ||--o{ central_lines : has
    patients ||--o{ clinical_measurements : has
    patients ||--o{ medical_history : has
    patients ||--o{ clabsi_events : has
    central_lines ||--o{ dressing_changes : tracks
    central_lines ||--o{ daily_line_checks : requires
    central_lines ||--o{ clabsi_events : develops

    patients {
        string patient_id PK
        int age
        string gender
        string race
        string ethnicity
        string unit_department
        datetime admission_date
        datetime discharge_date
        int length_of_stay
    }

    central_lines {
        string line_id PK
        string patient_id FK
        string line_type
        datetime insertion_date
        datetime removal_date
        boolean bundle_compliance
        int line_days
    }

    clinical_measurements {
        string measurement_id PK
        string patient_id FK
        datetime measurement_date
        decimal temperature
        decimal hemoglobin
        decimal wbc_count
        decimal neutrophil_count
    }

    medical_history {
        string history_id PK
        string patient_id FK
        string condition_type
        boolean condition_value
        datetime onset_date
    }

    clabsi_events {
        string event_id PK
        string patient_id FK
        string line_id FK
        datetime confirmation_date
        string organism
        string treatment_provided
        string outcome
    }

    dressing_changes {
        string change_id PK
        string line_id FK
        datetime change_date
        string performed_by
        string dressing_type
        string site_condition
    }

    daily_line_checks {
        string check_id PK
        string line_id FK
        datetime check_date
        string performed_by
        boolean site_clean
        boolean dressing_intact
        boolean chg_applied
    }
```
