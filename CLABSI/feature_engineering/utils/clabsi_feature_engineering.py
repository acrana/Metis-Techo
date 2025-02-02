# clabsi_feature_engineering.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, bindparam
import warnings
warnings.filterwarnings('ignore')


def connect_db():
    """Establish database connection to MIMIC IV"""
    try:
        engine = create_engine('****')
        print("Successfully connected to MIMIC IV database!")
        return engine
    except Exception as e:
        print(f"Connection failed: {e}")
        return None


def get_clabsi_cohort(engine):
    """
    Get validated CLABSI cases with enhanced organism classification
    and culture validation criteria.
    """
    clabsi_query = text("""
    WITH clabsi_icd AS (
        SELECT DISTINCT hadm_id, MIN(icd_code) as icd_code
        FROM mimiciv_hosp.diagnoses_icd
        WHERE icd_code IN ('99931', '99932', 'T80211A')
        GROUP BY hadm_id
    ),
    microbe_classifications AS (
        SELECT 
            m.hadm_id,
            m.charttime,
            m.org_name,
            COUNT(*) OVER (
                PARTITION BY m.hadm_id, m.org_name
            ) as cultures_of_org,
            COUNT(*) OVER (
                PARTITION BY m.hadm_id, 
                DATE_TRUNC('day', m.charttime),
                m.org_name
            ) as daily_org_cultures,
            CASE
                -- New microbe filter: exclude common contaminants unless >=2 cultures
                WHEN m.org_name IN ('Bacillus species', 'Micrococcus species')
                     AND COUNT(*) OVER (
                         PARTITION BY m.hadm_id, 
                         DATE_TRUNC('day', m.charttime),
                         m.org_name
                     ) < 2 THEN 'Contaminant'
                
                -- Rest of existing classification logic
                WHEN m.org_name IN (
                    'STAPH AUREUS COAG +', 'ESCHERICHIA COLI',
                    'KLEBSIELLA PNEUMONIAE', 'PSEUDOMONAS AERUGINOSA',
                    'ENTEROCOCCUS FAECIUM', 'CANDIDA ALBICANS'
                ) THEN 'Primary Pathogen'
                
                WHEN m.org_name IN (
                    'STAPHYLOCOCCUS, COAGULASE NEGATIVE',
                    'STAPHYLOCOCCUS EPIDERMIDIS'
                ) AND COUNT(*) OVER (
                    PARTITION BY m.hadm_id, 
                    DATE_TRUNC('day', m.charttime),
                    m.org_name
                ) >= 2 THEN 'Validated Common Contaminant'
                
                WHEN m.org_name LIKE 'CANDIDA%' OR
                     m.org_name IN (
                        'ENTEROCOCCUS FAECALIS',
                        'SERRATIA MARCESCENS',
                        'BACTEROIDES FRAGILIS GROUP'
                     ) THEN 'Other Pathogen'
                ELSE 'Requires Review'
            END as organism_classification
        FROM mimiciv_hosp.microbiologyevents m
        WHERE m.spec_type_desc = 'BLOOD CULTURE'
        AND m.org_name IS NOT NULL
    ),
    validated_cases AS (
        SELECT DISTINCT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.intime as icu_admission,
            i.outtime as icu_discharge,
            l.starttime as line_start,
            l.endtime as line_end,
            l.line_type,
            EXTRACT(EPOCH FROM (l.endtime - l.starttime))/86400 as line_duration_days,
            m.charttime as infection_time,
            m.org_name,
            m.cultures_of_org,
            m.daily_org_cultures,
            m.organism_classification
        FROM mimiciv_icu.icustays i
        INNER JOIN clabsi_icd c ON i.hadm_id = c.hadm_id
        INNER JOIN mimiciv_derived.invasive_line l ON i.stay_id = l.stay_id
        INNER JOIN microbe_classifications m ON i.hadm_id = m.hadm_id
        WHERE
            -- New line validation filter
            l.line_type IN (
                'PICC', 'Multi Lumen', 'Dialysis', 'Triple Introducer',
                'Pre-Sep', 'Hickman', 'Portacath', 'Cordis/Introducer',
                'Continuous Cardiac Output PA', 'PA'
            )
            AND l.line_type NOT IN ('Peripheral IV', 'Arterial Line')
            
            AND m.charttime > l.starttime
            AND m.charttime <= l.starttime + INTERVAL '14 days'
            AND EXTRACT(EPOCH FROM (l.endtime - l.starttime))/86400 >= 2
            AND m.organism_classification IN (
                'Primary Pathogen',
                'Validated Common Contaminant',
                'Other Pathogen'
            )
    )
    SELECT *,
        CASE
            WHEN organism_classification = 'Primary Pathogen' THEN 'Confirmed CLABSI'
            WHEN organism_classification = 'Validated Common Contaminant' 
                 AND daily_org_cultures >= 2 THEN 'Confirmed CLABSI'
            WHEN organism_classification = 'Other Pathogen' 
                 AND cultures_of_org >= 2 THEN 'Confirmed CLABSI'
            ELSE 'Requires Review'
        END as clabsi_status
    FROM validated_cases v
    WHERE line_duration_days >= 2
    ORDER BY stay_id, infection_time;
    """)
    
    clabsi_df = pd.read_sql(clabsi_query, engine)
    return clabsi_df


def get_control_cohort(engine, clabsi_cohort, matching_ratio=4):
    
    unique_stays = clabsi_cohort.drop_duplicates('stay_id')[['stay_id', 'line_type']]
    line_targets = unique_stays['line_type'].value_counts() * matching_ratio
    
    control_query = text("""
    WITH line_patients AS (
        SELECT DISTINCT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.intime as icu_admission,
            i.outtime as icu_discharge,
            l.starttime as line_start,
            l.endtime as line_end,
            l.line_type,
            EXTRACT(EPOCH FROM (l.endtime - l.starttime))/86400 as line_duration_days,
            p.gender,
            p.anchor_age + EXTRACT(EPOCH FROM adm.admittime - MAKE_TIMESTAMP(p.anchor_year, 1, 1, 0, 0, 0))/31556908.8 AS admission_age,
            adm.race,
            adm.admission_type
        FROM mimiciv_icu.icustays i
        INNER JOIN mimiciv_derived.invasive_line l ON i.stay_id = l.stay_id
        INNER JOIN mimiciv_hosp.admissions adm ON i.hadm_id = adm.hadm_id
        INNER JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
        WHERE l.line_type IN (
            'PICC', 'Multi Lumen', 'Dialysis', 'Triple Introducer',
            'Pre-Sep', 'Hickman', 'Portacath', 'Cordis/Introducer',
            'Continuous Cardiac Output PA', 'PA'
        )
        AND EXTRACT(EPOCH FROM (l.endtime - l.starttime))/86400 >= 2
    ),
    excluded_patients AS (
        SELECT DISTINCT p.stay_id
        FROM line_patients p
        LEFT JOIN mimiciv_hosp.diagnoses_icd d ON p.hadm_id = d.hadm_id
        LEFT JOIN mimiciv_hosp.microbiologyevents m ON p.hadm_id = m.hadm_id
            AND m.charttime BETWEEN p.line_start AND p.line_end
        WHERE d.icd_code IN ('99931', '99932', 'T80211A')
        OR (
            m.spec_type_desc = 'BLOOD CULTURE'
            AND m.org_name IS NOT NULL
        )
    ),
    eligible_controls AS (
        SELECT 
            p.*,
            CASE 
                WHEN p.line_type = 'Multi Lumen' THEN :multi_lumen_target
                WHEN p.line_type = 'Dialysis' THEN :dialysis_target
                WHEN p.line_type = 'PICC' THEN :picc_target
                WHEN p.line_type = 'PA' THEN :pa_target
                WHEN p.line_type = 'Cordis/Introducer' THEN :cordis_target
                WHEN p.line_type = 'Hickman' THEN :hickman_target
                WHEN p.line_type = 'Portacath' THEN :portacath_target
                WHEN p.line_type = 'Continuous Cardiac Output PA' THEN :cco_pa_target
                ELSE 40
            END as target_n,
            ROW_NUMBER() OVER (
                PARTITION BY p.line_type 
                ORDER BY random()
            ) as type_rank
        FROM line_patients p
        WHERE NOT EXISTS (
            SELECT 1 
            FROM excluded_patients e 
            WHERE p.stay_id = e.stay_id
        )
    )
    SELECT *
    FROM eligible_controls
    WHERE type_rank <= target_n
    ORDER BY line_type, type_rank;
    """)
    
    controls_df = pd.read_sql(
        control_query, 
        engine,
        params={
            'multi_lumen_target': int(line_targets.get('Multi Lumen', 0)),
            'dialysis_target': int(line_targets.get('Dialysis', 0)),
            'picc_target': int(line_targets.get('PICC', 0)),
            'pa_target': int(line_targets.get('PA', 0)),
            'cordis_target': int(line_targets.get('Cordis/Introducer', 0)),
            'hickman_target': int(line_targets.get('Hickman', 0)),
            'portacath_target': int(line_targets.get('Portacath', 0)),
            'cco_pa_target': int(line_targets.get('Continuous Cardiac Output PA', 0))
        }
    )
    
    print("\nControl Cohort Summary:")
    print("-" * 30)
    print(f"Total controls: {len(controls_df)}")
    print(f"Unique patients: {controls_df['subject_id'].nunique()}")
    print(f"Unique ICU stays: {controls_df['stay_id'].nunique()}")
    
    print("\nMatching Ratios by Line Type:")
    unique_clabsi = unique_stays['line_type'].value_counts()
    unique_controls = controls_df['line_type'].value_counts()
    ratios = pd.DataFrame({
        'CLABSI_stays': unique_clabsi,
        'Control_stays': unique_controls,
        'Ratio': unique_controls / unique_clabsi
    }).round(2)
    print(ratios)
    
    return controls_df


def get_demographic_features(engine, cohort_df):
    """
    Extract basic demographic and comorbidity features for CLABSI prediction.
    """
    demo_query = text("""
    WITH demographics AS (
        SELECT DISTINCT
            icu.stay_id,
            icu.hadm_id,
            icu.subject_id,
            p.gender,
            p.anchor_age + EXTRACT(EPOCH FROM adm.admittime - MAKE_TIMESTAMP(p.anchor_year, 1, 1, 0, 0, 0))/31556908.8 AS admission_age,
            CASE
                WHEN adm.race LIKE '%BLACK%' OR adm.race LIKE '%AFRICAN%' THEN 'BLACK'
                WHEN adm.race LIKE '%ASIAN%' THEN 'ASIAN'
                WHEN adm.race LIKE '%HISPANIC%' OR adm.race LIKE '%LATINO%' THEN 'HISPANIC'
                WHEN adm.race LIKE '%WHITE%' THEN 'WHITE'
                ELSE 'OTHER'
            END AS ethnicity
        FROM mimiciv_icu.icustays icu
        INNER JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id
        INNER JOIN mimiciv_hosp.patients p ON icu.subject_id = p.subject_id
        WHERE icu.stay_id IN :stay_ids
    ),
    comorbidities AS (
        SELECT 
            icu.stay_id,
            MAX(CASE WHEN d.icd_code LIKE 'I10%' OR d.icd_code LIKE '401%' THEN 1 ELSE 0 END) as has_htn,
            MAX(CASE WHEN d.icd_code LIKE 'J44%' OR d.icd_code LIKE '496%' THEN 1 ELSE 0 END) as has_copd,
            MAX(CASE WHEN d.icd_code LIKE 'K70%' OR d.icd_code LIKE 'K72%' OR d.icd_code LIKE 'K76%' THEN 1 ELSE 0 END) as has_liver,
            MAX(CASE WHEN d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%' OR d.icd_code LIKE '410%' THEN 1 ELSE 0 END) as has_mi,
            MAX(CASE WHEN d.icd_code LIKE 'I50%' OR d.icd_code LIKE '428%' THEN 1 ELSE 0 END) as has_chf,
            MAX(CASE WHEN d.icd_code LIKE 'I63%' OR d.icd_code LIKE '433%' OR d.icd_code LIKE '434%' THEN 1 ELSE 0 END) as has_cva,
            MAX(CASE WHEN d.icd_code LIKE 'C%' OR d.icd_code LIKE '14%' OR d.icd_code LIKE '20%' THEN 1 ELSE 0 END) as has_cancer,
            MAX(CASE WHEN d.icd_code LIKE 'B20%' OR d.icd_code LIKE 'B21%' OR d.icd_code LIKE '042%' THEN 1 ELSE 0 END) as has_aids,
            MAX(CASE WHEN d.icd_code LIKE 'E11%' OR d.icd_code LIKE '250%' THEN 1 ELSE 0 END) as has_diabetes
        FROM mimiciv_icu.icustays icu
        LEFT JOIN mimiciv_hosp.diagnoses_icd d ON icu.hadm_id = d.hadm_id
        WHERE icu.stay_id IN :stay_ids
        GROUP BY icu.stay_id
    )
    SELECT 
        d.*,
        COALESCE(c.has_htn, 0) as has_htn,
        COALESCE(c.has_copd, 0) as has_copd,
        COALESCE(c.has_liver, 0) as has_liver,
        COALESCE(c.has_mi, 0) as has_mi,
        COALESCE(c.has_chf, 0) as has_chf,
        COALESCE(c.has_cva, 0) as has_cva,
        COALESCE(c.has_cancer, 0) as has_cancer,
        COALESCE(c.has_aids, 0) as has_aids,
        COALESCE(c.has_diabetes, 0) as has_diabetes
    FROM demographics d
    LEFT JOIN comorbidities c ON d.stay_id = c.stay_id
    ORDER BY d.stay_id;
    """).bindparams(bindparam("stay_ids", expanding=True))
    
    stay_ids = tuple(int(x) for x in cohort_df['stay_id'].unique())
    demo_df = pd.read_sql(demo_query, engine, params={'stay_ids': stay_ids})
    
    demo_df['gender'] = (demo_df['gender'] == 'M').astype(int)
    demo_df = pd.get_dummies(demo_df, columns=['ethnicity'], prefix='ethnicity')
    
    print("\nDemographic Features Summary:")
    print("-" * 30)
    print(f"Features extracted for {len(demo_df)} cases")
    print("\nNumerical Feature Statistics:")
    print(demo_df[['admission_age']].describe().round(2))
    print("\nComorbidity Frequencies:")
    comorbidity_cols = [col for col in demo_df.columns if col.startswith('has_')]
    print(demo_df[comorbidity_cols].sum().sort_values(ascending=False))
    
    return demo_df


def get_device_features(engine, cohort_df):
    """
    Extract tracheostomy and ostomy features for CLABSI prediction.
    """
    device_query = text("""
    WITH device_timing AS (
       SELECT 
           icu.stay_id,
           icu.hadm_id,
           icu.subject_id,
           COALESCE(
               (SELECT MIN(charttime) 
                FROM mimiciv_hosp.microbiologyevents me 
                WHERE me.hadm_id = icu.hadm_id 
                  AND me.org_name IS NOT NULL 
                  AND me.spec_type_desc = 'BLOOD CULTURE'),
               icu.outtime
           ) as end_time
       FROM mimiciv_icu.icustays icu
       WHERE icu.stay_id IN :stay_ids
    ),
    trach_status AS (
       SELECT DISTINCT
           dt.stay_id,
           MAX(CASE WHEN 
               LOWER(ce.value) IN ('tracheostomy tube', 'trach mask')
               AND ce.charttime < dt.end_time
           THEN 1 ELSE 0 END) as has_tracheostomy
       FROM device_timing dt
       LEFT JOIN mimiciv_icu.chartevents ce ON dt.stay_id = ce.stay_id
       LEFT JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
       WHERE (LOWER(di.label) LIKE '%o2%' OR LOWER(di.label) LIKE '%delivery%')
       GROUP BY dt.stay_id
    ),
    ostomy_status AS (
        SELECT 
            dt.stay_id,
            MAX(CASE WHEN 
                ce.itemid IN (
                    227458,
                    227637,
                    228341,
                    228342,
                    228343,
                    228344,
                    228345,
                    228346,
                    228347
                )
                AND ce.charttime < dt.end_time
            THEN 1 ELSE 0 END) as has_ostomy
        FROM device_timing dt
        LEFT JOIN mimiciv_icu.chartevents ce ON dt.stay_id = ce.stay_id
        GROUP BY dt.stay_id
    )
    SELECT 
        dt.stay_id,
        COALESCE(t.has_tracheostomy, 0) as has_tracheostomy,
        COALESCE(o.has_ostomy, 0) as has_ostomy
    FROM device_timing dt
    LEFT JOIN trach_status t ON dt.stay_id = t.stay_id
    LEFT JOIN ostomy_status o ON dt.stay_id = o.stay_id
    ORDER BY dt.stay_id;
    """).bindparams(bindparam("stay_ids", expanding=True))
   
    stay_ids = tuple(int(x) for x in cohort_df['stay_id'].unique())
    device_df = pd.read_sql(device_query, engine, params={'stay_ids': stay_ids})
   
    print("\nDevice Features Summary:")
    print("-" * 30)
    print(f"Features extracted for {len(device_df)} cases")
    print("\nDevice Frequencies:")
    device_cols = ['has_tracheostomy', 'has_ostomy']
    print(device_df[device_cols].sum().sort_values(ascending=False))
   
    return device_df


def get_line_care(engine, cohort_df):
    """Extract central line care metrics."""
    line_care_query = text("""
    WITH line_events AS (
        SELECT 
            ie.stay_id,
            ie.hadm_id,
            il.line_type,
            il.line_site,
            CASE 
                WHEN LOWER(il.line_site) LIKE '%femoral%' THEN 'Femoral'
                WHEN LOWER(il.line_site) LIKE '%subclavian%' THEN 'Subclavian'
                WHEN LOWER(il.line_site) LIKE '%jugular%' OR LOWER(il.line_site) LIKE '%ij%' THEN 'Internal Jugular'
                WHEN LOWER(il.line_site) LIKE '%picc%' OR LOWER(il.line_site) LIKE '%peripherally inserted%' THEN 'PICC'
                ELSE 'Other'
            END as site_category,
            CASE 
                WHEN LOWER(il.line_site) LIKE '%femoral%' THEN 'High Risk'
                WHEN LOWER(il.line_site) LIKE '%subclavian%' THEN 'Low Risk'
                WHEN LOWER(il.line_site) LIKE '%jugular%' OR LOWER(il.line_site) LIKE '%ij%' THEN 'Moderate Risk'
                ELSE 'Standard Risk'
            END as risk_category,
            il.starttime,
            il.endtime,
            EXTRACT(EPOCH FROM (il.endtime - il.starttime))/86400.0 as line_duration_days,
            LEAD(il.starttime) OVER (
                PARTITION BY ie.stay_id, il.line_type 
                ORDER BY il.starttime
            ) as next_line_start,
            LAG(il.endtime) OVER (
                PARTITION BY ie.stay_id, il.line_type 
                ORDER BY il.starttime
            ) as prev_line_end
        FROM mimiciv_icu.icustays ie
        INNER JOIN mimiciv_derived.invasive_line il ON ie.stay_id = il.stay_id
        WHERE ie.stay_id IN :stay_ids
        AND il.line_type IN (
            'PICC', 'Multi Lumen', 'Dialysis', 'Triple Introducer',
            'Pre-Sep', 'Hickman', 'Portacath', 'Cordis/Introducer',
            'Continuous Cardiac Output PA', 'PA'
        )
    ),
    site_complications AS (
        SELECT 
            ce.stay_id,
            il.line_type,
            il.line_site,
            COUNT(CASE WHEN LOWER(ce.value) LIKE '%redness%' OR LOWER(ce.value) LIKE '%erythema%' THEN 1 END) as redness_count,
            COUNT(CASE WHEN LOWER(ce.value) LIKE '%drainage%' OR LOWER(ce.value) LIKE '%purulent%' OR LOWER(ce.value) LIKE '%exudate%' THEN 1 END) as drainage_count,
            COUNT(CASE WHEN LOWER(ce.value) LIKE '%tender%' OR LOWER(ce.value) LIKE '%pain%' THEN 1 END) as tenderness_count,
            COUNT(CASE WHEN LOWER(ce.value) LIKE '%swelling%' OR LOWER(ce.value) LIKE '%edema%' THEN 1 END) as swelling_count,
            COUNT(CASE WHEN LOWER(ce.value) LIKE '%blood%' OR LOWER(ce.value) LIKE '%oozing%' THEN 1 END) as bleeding_count,
            COUNT(CASE WHEN LOWER(ce.value) LIKE '%infiltrat%' OR LOWER(ce.value) LIKE '%migration%' OR LOWER(ce.value) LIKE '%dislodg%' THEN 1 END) as severe_complication_count
        FROM mimiciv_derived.invasive_line il
        INNER JOIN mimiciv_icu.chartevents ce ON il.stay_id = ce.stay_id
            AND ce.charttime BETWEEN il.starttime AND il.endtime
        WHERE ce.itemid IN (
            224188, 224269, 224267, 224270, 224263, 224264, 224268, 225315
        )
        AND il.line_type IN (
            'PICC', 'Multi Lumen', 'Dialysis', 'Triple Introducer',
            'Pre-Sep', 'Hickman', 'Portacath', 'Cordis/Introducer',
            'Continuous Cardiac Output PA', 'PA'
        )
        AND ce.stay_id IN :stay_ids
        GROUP BY ce.stay_id, il.line_type, il.line_site
    )
    SELECT 
        le.stay_id,
        le.line_type,
        le.site_category,
        le.risk_category,
        COUNT(DISTINCT le.line_site) as unique_sites,
        SUM(CASE WHEN le.site_category = 'Femoral' THEN 1 ELSE 0 END) as femoral_count,
        SUM(CASE WHEN le.site_category = 'Subclavian' THEN 1 ELSE 0 END) as subclavian_count,
        SUM(CASE WHEN le.site_category = 'Internal Jugular' THEN 1 ELSE 0 END) as ij_count,
        SUM(CASE WHEN le.site_category = 'PICC' THEN 1 ELSE 0 END) as picc_count,
        MIN(le.line_duration_days) as shortest_duration,
        MAX(le.line_duration_days) as longest_duration,
        AVG(le.line_duration_days) as avg_duration,
        COALESCE(MAX(sc.redness_count), 0) as total_redness_events,
        COALESCE(MAX(sc.drainage_count), 0) as total_drainage_events,
        COALESCE(MAX(sc.tenderness_count), 0) as total_tenderness_events,
        COALESCE(MAX(sc.swelling_count), 0) as total_swelling_events,
        COALESCE(MAX(sc.bleeding_count), 0) as total_bleeding_events,
        COALESCE(MAX(sc.severe_complication_count), 0) as total_severe_complications
    FROM line_events le
    LEFT JOIN site_complications sc ON le.stay_id = sc.stay_id 
        AND le.line_type = sc.line_type
        AND le.line_site = sc.line_site
    GROUP BY 
        le.stay_id,
        le.line_type,
        le.site_category,
        le.risk_category
    ORDER BY 
        le.stay_id, 
        le.line_type;
    """).bindparams(bindparam("stay_ids", expanding=True))
    
    stay_ids = tuple(int(x) for x in cohort_df['stay_id'].unique())
    line_care_df = pd.read_sql(line_care_query, engine, params={'stay_ids': stay_ids})
    
    print("\nCentral Line Care Metrics Summary:")
    print("-" * 40)
    print(f"Total unique stays: {line_care_df['stay_id'].nunique()}")
    print(f"Line types captured: {line_care_df['line_type'].nunique()}")
    print("\nRisk Category Distribution:")
    print(line_care_df['risk_category'].value_counts(dropna=False))
    
    return line_care_df


def get_lab_values(engine, cohort_df):
    """
    Extract laboratory values with consistent columns across all panels.
    """
    lab_query = text("""
    WITH base_labs AS (
        SELECT 
            ie.stay_id,
            cb.charttime,
            'CBC' as panel,
            cb.wbc,
            cb.hemoglobin,
            cb.platelet,
            NULL::double precision as creatinine,
            NULL::double precision as sodium,
            NULL::double precision as potassium,
            NULL::double precision as chloride,
            NULL::double precision as bicarbonate,
            NULL::double precision as inr,
            NULL::double precision as pt,
            NULL::double precision as ptt
        FROM mimiciv_icu.icustays ie
        INNER JOIN mimiciv_derived.complete_blood_count cb ON ie.subject_id = cb.subject_id
        WHERE ie.stay_id IN :stay_ids
            
        UNION ALL
        
        SELECT 
            ie.stay_id,
            ch.charttime,
            'CHEM' as panel,
            NULL::double precision as wbc,
            NULL::double precision as hemoglobin,
            NULL::double precision as platelet,
            ch.creatinine,
            ch.sodium,
            ch.potassium,
            ch.chloride,
            ch.bicarbonate,
            NULL::double precision as inr,
            NULL::double precision as pt,
            NULL::double precision as ptt
        FROM mimiciv_icu.icustays ie
        INNER JOIN mimiciv_derived.chemistry ch ON ie.subject_id = ch.subject_id
        WHERE ie.stay_id IN :stay_ids
            
        UNION ALL
        
        SELECT 
            ie.stay_id,
            cg.charttime,
            'COAG' as panel,
            NULL::double precision as wbc,
            NULL::double precision as hemoglobin,
            NULL::double precision as platelet,
            NULL::double precision as creatinine,
            NULL::double precision as sodium,
            NULL::double precision as potassium,
            NULL::double precision as chloride,
            NULL::double precision as bicarbonate,
            cg.inr,
            cg.pt,
            cg.ptt
        FROM mimiciv_icu.icustays ie
        INNER JOIN mimiciv_derived.coagulation cg ON ie.subject_id = cg.subject_id
        WHERE ie.stay_id IN :stay_ids
    )
    SELECT 
        stay_id,
        MIN(wbc) as wbc_min,
        MAX(wbc) as wbc_max,
        AVG(wbc) as wbc_mean,
        MIN(platelet) as plt_min,
        MAX(platelet) as plt_max,
        AVG(platelet) as plt_mean,
        MIN(hemoglobin) as hgb_min,
        MAX(hemoglobin) as hgb_max,
        AVG(hemoglobin) as hgb_mean,
        MIN(creatinine) as creat_min,
        MAX(creatinine) as creat_max,
        AVG(creatinine) as creat_mean,
        MIN(sodium) as sodium_min,
        MAX(sodium) as sodium_max,
        AVG(sodium) as sodium_mean,
        MIN(inr) as inr_min,
        MAX(inr) as inr_max,
        AVG(inr) as inr_mean,
        MIN(pt) as pt_min,
        MAX(pt) as pt_max,
        AVG(pt) as pt_mean,
        COUNT(DISTINCT charttime) as lab_count,
        SUM(CASE WHEN wbc > 12 THEN 1 ELSE 0 END) as high_wbc_count,
        SUM(CASE WHEN wbc < 4 THEN 1 ELSE 0 END) as low_wbc_count,
        SUM(CASE WHEN platelet < 150 THEN 1 ELSE 0 END) as low_plt_count
    FROM base_labs
    GROUP BY stay_id
    ORDER BY stay_id;
    """).bindparams(bindparam("stay_ids", expanding=True))
    
    stay_ids = [int(x) for x in cohort_df['stay_id'].unique()]
    lab_df = pd.read_sql(lab_query, engine, params={'stay_ids': stay_ids})
    
    if len(lab_df) > 0:
        print("\nLaboratory Values Summary:")
        print("-" * 30)
        print(f"Features extracted for {len(lab_df)} cases")
        if len(lab_df) > 1:
            print("\nLab Measurement Frequencies:")
            print(lab_df['lab_count'].describe().round(2))
            print("\nKey Lab Value Statistics:")
            key_metrics = ['wbc_mean', 'plt_mean', 'hgb_mean',
                           'high_wbc_count', 'low_plt_count']
            print(lab_df[key_metrics].describe().round(2))
    
    return lab_df


def get_all_severity_scores(engine, cohort_df):
    """
    Retrieve first-day SOFA, APS-III, and SAPS-II scores for CLABSI patients.
    """
    queries = {
        'sofa': text("""
        SELECT
            stay_id,
            sofa AS sofa_score,
            respiration,
            coagulation,
            liver,
            cardiovascular,
            cns,
            renal
        FROM mimiciv_derived.first_day_sofa
        WHERE stay_id IN :stay_ids
        """).bindparams(bindparam("stay_ids", expanding=True)),
        'apsiii': text("""
        WITH apsiii_cte AS (
            SELECT
                ie.subject_id,
                ie.hadm_id,
                ie.stay_id,
                apsiii,
                CAST(1 AS DOUBLE PRECISION) / (
                    1 + EXP(-(-4.4360 + 0.04726 * (apsiii)))
                ) AS apsiii_prob
            FROM mimiciv_icu.icustays AS ie
            LEFT JOIN mimiciv_derived.apsiii AS apsiii_table
            ON ie.stay_id = apsiii_table.stay_id
        )
        SELECT * FROM apsiii_cte
        WHERE stay_id IN :stay_ids;
        """).bindparams(bindparam("stay_ids", expanding=True)),
        'sapsii': text("""
        WITH sapsii_cte AS (
            SELECT
                s.subject_id,
                s.hadm_id,
                s.stay_id,
                sapsii,
                CAST(1 AS DOUBLE PRECISION) / (
                    1 + EXP(-(-7.7631 + 0.0737 * (sapsii) + 0.9971 * LN(sapsii + 1)))
                ) AS sapsii_prob
            FROM mimiciv_derived.sapsii AS s
        )
        SELECT * FROM sapsii_cte
        WHERE stay_id IN :stay_ids;
        """).bindparams(bindparam("stay_ids", expanding=True))
    }

    stay_ids = [int(x) for x in cohort_df['stay_id'].unique()]
    results = {}

    for score_type, query in queries.items():
        try:
            print(f"Retrieving {score_type.upper()} scores...")
            results[score_type] = pd.read_sql(query, engine, params={'stay_ids': stay_ids})
            print(f"Successfully retrieved {len(results[score_type])} rows for {score_type.upper()} scores.")
        except Exception as e:
            print(f"Error retrieving {score_type.upper()} scores: {e}")
            results[score_type] = None

    return results


def get_line_type_risk_analysis(clabsi_cohort):
    """
    Analyze total line duration instead of duration before CLABSI.
    """
    clabsi_cohort['line_duration_total'] = (
        (clabsi_cohort['line_end'] - clabsi_cohort['line_start']).dt.total_seconds() / 86400
    )
    clabsi_cohort = clabsi_cohort[clabsi_cohort['line_duration_total'] > 0]

    line_risk_summary = clabsi_cohort.groupby('line_type').agg(
        avg_duration=('line_duration_total', 'mean'),
        median_duration=('line_duration_total', 'median'),
        max_duration=('line_duration_total', 'max'),
        line_count=('line_type', 'count')
    ).reset_index()
    
    return line_risk_summary


def threshold_analysis(clabsi_cohort, thresholds=[3, 5, 7, 10]):
    """
    Perform threshold analysis for line durations.
    """
    if 'line_duration_before_clabsi' not in clabsi_cohort.columns:
        clabsi_cohort['adjusted_endtime'] = clabsi_cohort[['line_end', 'infection_time']].min(axis=1)
        clabsi_cohort['line_duration_before_clabsi'] = (
            (clabsi_cohort['adjusted_endtime'] - clabsi_cohort['line_start']).dt.total_seconds() / 86400
        )
        clabsi_cohort = clabsi_cohort[clabsi_cohort['line_duration_before_clabsi'] > 0]

    results = []
    for line_type, group in clabsi_cohort.groupby('line_type'):
        total_lines = len(group)
        for threshold in thresholds:
            exceed_count = (group['line_duration_before_clabsi'] > threshold).sum()
            exceed_rate = exceed_count / total_lines * 100
            results.append({
                'line_type': line_type,
                'threshold': threshold,
                'exceed_count': exceed_count,
                'total_lines': total_lines,
                'exceed_rate': exceed_rate
            })
    threshold_df = pd.DataFrame(results)
    return threshold_df


def get_chg_bath_features(engine, cohort_df):
    """CHG bath analysis."""
    bath_query = text("""
    WITH bath_events AS (
        SELECT 
            ce.stay_id,
            ce.charttime,
            EXTRACT(EPOCH FROM (
                ce.charttime - LAG(ce.charttime) OVER (
                    PARTITION BY ce.stay_id ORDER BY ce.charttime
                )
            ))/86400 AS days_since_last_bath
        FROM mimiciv_icu.chartevents ce
        INNER JOIN mimiciv_derived.invasive_line il ON ce.stay_id = il.stay_id
            AND ce.charttime BETWEEN il.starttime - INTERVAL '2 hours' AND il.endtime + INTERVAL '2 hours'
        WHERE ce.itemid = 228137
    ),
    bath_analysis AS (
        SELECT
            stay_id,
            COUNT(*) AS total_baths,
            MAX(days_since_last_bath) AS max_gap_between_baths
        FROM bath_events
        GROUP BY stay_id
    )
    SELECT 
        ba.*,
        CASE WHEN ba.total_baths = 0 THEN 1 ELSE 0 END as no_bath_flag,
        CASE WHEN ba.max_gap_between_baths > 3 THEN 1 ELSE 0 END as irregular_bathing
    FROM bath_analysis ba
    WHERE ba.stay_id IN :stay_ids
    """).bindparams(bindparam("stay_ids", expanding=True))
    
    stay_ids = tuple(int(x) for x in cohort_df['stay_id'].unique())
    return pd.read_sql(bath_query, engine, params={'stay_ids': stay_ids})


def get_medications(engine, cohort_df):
    """
    Extract medication features focusing on high-risk medications.
    """
    med_query = text("""
    WITH antibiotics AS (
        SELECT 
            stay_id,
            COUNT(DISTINCT antibiotic) AS distinct_antibiotics,
            COUNT(*) AS total_antibiotic_orders,
            MAX(EXTRACT(EPOCH FROM (stoptime - starttime))/86400) AS max_antibiotic_duration
        FROM mimiciv_derived.antibiotic
        WHERE stay_id IN :stay_ids
        GROUP BY stay_id
    ),
    tpn AS (
        SELECT 
            stay_id,
            COUNT(*) AS tpn_orders,
            SUM(CASE WHEN rate > 0 THEN 1 ELSE 0 END) AS tpn_days
        FROM mimiciv_icu.inputevents
        WHERE stay_id IN :stay_ids
        AND itemid IN (226089, 227690)
        GROUP BY stay_id
    ),
    high_risk_meds AS (
        SELECT 
            stay_id,
            COUNT(CASE WHEN itemid IN (221662, 221653, 221668) THEN 1 ELSE NULL END) AS high_risk_med_count,
            COUNT(DISTINCT itemid) AS distinct_high_risk_meds
        FROM mimiciv_icu.inputevents
        WHERE stay_id IN :stay_ids
        GROUP BY stay_id
    )
    SELECT 
        COALESCE(a.stay_id, t.stay_id, h.stay_id) AS stay_id,
        COALESCE(distinct_antibiotics, 0) AS distinct_antibiotics,
        COALESCE(total_antibiotic_orders, 0) AS total_antibiotic_orders,
        COALESCE(max_antibiotic_duration, 0) AS max_antibiotic_duration,
        COALESCE(tpn_orders, 0) AS tpn_orders,
        COALESCE(tpn_days, 0) AS tpn_days,
        COALESCE(high_risk_med_count, 0) AS high_risk_med_count,
        COALESCE(distinct_high_risk_meds, 0) AS distinct_high_risk_meds
    FROM antibiotics a
    FULL OUTER JOIN tpn t ON a.stay_id = t.stay_id
    FULL OUTER JOIN high_risk_meds h ON a.stay_id = h.stay_id
    """).bindparams(bindparam("stay_ids", expanding=True))
    
    stay_ids = [int(x) for x in cohort_df['stay_id'].unique()]
    medication_df = pd.read_sql(med_query, engine, params={'stay_ids': stay_ids})
    
    print("\nMedication Features Retrieved:")
    print(f"Number of rows: {len(medication_df)}")
    print(medication_df.head())
    print("\nSummary Statistics:")
    print(medication_df.describe().round(2))
    
    return medication_df


def get_clinical_features(engine, cohort_df):
    """Extract central line care metrics with CHG adherence."""
    line_care_query = text("""
    WITH line_events AS (
        SELECT 
            ie.stay_id,
            il.line_type,
            il.starttime as line_start,
            il.endtime as line_end,
            MAX(ce.charttime) as last_dressing_change,
            COUNT(CASE WHEN ce.itemid = 228137 THEN 1 END) as actual_baths,
            CEIL(EXTRACT(EPOCH FROM (il.endtime - il.starttime))/86400/2) as expected_baths
        FROM mimiciv_icu.icustays ie
        INNER JOIN mimiciv_derived.invasive_line il ON ie.stay_id = il.stay_id
        LEFT JOIN mimiciv_icu.chartevents ce ON ie.stay_id = ce.stay_id
            AND ce.charttime BETWEEN il.starttime AND il.endtime
        WHERE ie.stay_id IN :stay_ids
        GROUP BY ie.stay_id, il.line_type, il.starttime, il.endtime
    ),
    infection_timing AS (
        SELECT 
            i.stay_id,
            MIN(m.charttime) as infection_time
        FROM mimiciv_hosp.microbiologyevents m
        INNER JOIN mimiciv_icu.icustays i ON m.hadm_id = i.hadm_id
            AND m.charttime BETWEEN i.intime AND i.outtime
        WHERE m.spec_type_desc = 'BLOOD CULTURE'
          AND m.org_name IS NOT NULL
        GROUP BY i.stay_id
    )
    SELECT 
        le.stay_id,
        le.line_type,
        EXTRACT(EPOCH FROM (it.infection_time - le.last_dressing_change))/86400 as days_since_last_dressing_change,
        le.actual_baths / NULLIF(le.expected_baths, 0) as chg_adherence_ratio,
        EXTRACT(EPOCH FROM (le.line_end - le.line_start))/86400 as line_duration_days
    FROM line_events le
    INNER JOIN infection_timing it ON le.stay_id = it.stay_id
    ORDER BY le.stay_id;
    """).bindparams(bindparam("stay_ids", expanding=True))
    
    stay_ids = tuple(int(x) for x in cohort_df['stay_id'].unique())
    line_care_df = pd.read_sql(line_care_query, engine, params={'stay_ids': stay_ids})
    return line_care_df


def analyze_line_sites_clabsi_risk(engine, cohort_df, line_care_df):
    """
    Analyze which central line sites are associated with higher CLABSI risk,
    excluding arterial lines.
    """
    site_query = text("""
    WITH line_infections AS (
        SELECT 
            il.stay_id,
            il.line_type,
            il.line_site,
            CASE WHEN il.stay_id IN :clabsi_ids THEN 1 ELSE 0 END as is_clabsi,
            EXTRACT(EPOCH FROM (il.endtime - il.starttime))/86400 as line_duration_days
        FROM mimiciv_derived.invasive_line il
        WHERE il.stay_id IN :clabsi_ids
        AND il.line_site IS NOT NULL
        AND il.line_type NOT LIKE 'Arterial'
        AND il.line_type IN (
            'PICC', 'Multi Lumen', 'Dialysis', 'Triple Introducer', 
            'Pre-Sep', 'Hickman', 'Portacath', 'Cordis/Introducer',
            'Continuous Cardiac Output PA', 'PA'
        )
    )
    SELECT 
        line_type,
        line_site,
        COUNT(*) as total_lines,
        SUM(is_clabsi) as clabsi_count,
        ROUND(AVG(line_duration_days), 2) as avg_duration_days
    FROM line_infections
    GROUP BY line_type, line_site
    HAVING COUNT(*) >= 5
    ORDER BY clabsi_count DESC;
    """).bindparams(bindparam("clabsi_ids", expanding=True))
    
    try:
        clabsi_ids = [int(x) for x in cohort_df['stay_id'].unique()]
        site_risk = pd.read_sql(site_query, engine, params={'clabsi_ids': clabsi_ids})
        print("\nCentral Line Sites Associated with CLABSI:")
        print("-" * 40)
        print(site_risk)
        return site_risk
    except Exception as e:
        print(f"Error executing analysis: {e}")
        return None


class CLABSIFeatureProcessor:
    """
    Unified feature processing pipeline for CLABSI prediction.
    """
    def __init__(self, engine):
        self.engine = engine
        self.time_windows = {
            'short': '24 hours',
            'medium': '48 hours',
            'long': '72 hours'
        }
        self.duration_thresholds = [3, 5, 7, 10]
        try:
            query = "SELECT stay_id FROM mimiciv_icu.icustays LIMIT 1"
            self.example_stay_id = pd.read_sql(query, self.engine)['stay_id'].iloc[0]
        except Exception as e:
            self.example_stay_id = None
            print(f"Warning: Could not set example_stay_id: {e}")
    
    def _get_infection_time(self, stay_id):
        query = text("""
            SELECT MIN(m.charttime) AS infection_time
            FROM mimiciv_hosp.microbiologyevents m
            INNER JOIN mimiciv_icu.icustays i ON m.hadm_id = i.hadm_id
            WHERE i.stay_id = :stay_id
            AND m.spec_type_desc = 'BLOOD CULTURE'
            AND m.org_name IS NOT NULL
        """)
        df = pd.read_sql(query, self.engine, params={'stay_id': stay_id})
        if pd.notnull(df['infection_time'].iloc[0]):
            return df['infection_time'].iloc[0]
        else:
            outtime_query = text("SELECT outtime FROM mimiciv_icu.icustays WHERE stay_id = :stay_id")
            outtime_df = pd.read_sql(outtime_query, self.engine, params={'stay_id': stay_id})
            return outtime_df['outtime'].iloc[0]
    
    def get_all_features(self, stay_id):
        try:
            stay_df = pd.DataFrame({'stay_id': [stay_id]})
            infection_time = self._get_infection_time(stay_id)
            stay_df['infection_time'] = infection_time
            
            features = {
                'demographics': get_demographic_features(self.engine, stay_df),
                'devices': get_device_features(self.engine, stay_df),
                'labs': get_lab_values(self.engine, stay_df),
                'severity': get_all_severity_scores(self.engine, stay_df),
                'line_care': get_line_care(self.engine, stay_df),
                'medications': get_medications(self.engine, stay_df),
                'clinical': get_clinical_features(self.engine, stay_df),
                'chg_bath': get_chg_bath_features(self.engine, stay_df)
            }
            
            for key, value in features.items():
                setattr(self, f"{key}_features", value)
            
            combined_features = self._combine_features(features)
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features for stay_id {stay_id}: {str(e)}")
            return None
            
    def get_batch_features(self, cohort_df):
        try:
            print(f"Processing {len(cohort_df)} stays...")
            features = {
                'demographics': get_demographic_features(self.engine, cohort_df),
                'devices': get_device_features(self.engine, cohort_df),
                'labs': get_lab_values(self.engine, cohort_df),
                'severity': get_all_severity_scores(self.engine, cohort_df),
                'line_care': get_line_care(self.engine, cohort_df),
                'medications': get_medications(self.engine, cohort_df),
                'clinical': get_clinical_features(self.engine, cohort_df),
                'chg_bath': get_chg_bath_features(self.engine, cohort_df)
            }
            
            for name, feature_set in features.items():
                if feature_set is not None:
                    print(f"{name}: {len(feature_set)} rows")
                else:
                    print(f"{name}: None")
            
            for key, value in features.items():
                setattr(self, f"{key}_features", value)
            
            print("Combining features...")
            combined_features = self._combine_features(features)
            if combined_features is not None:
                cohort_cols = ['line_type', 'line_site', 'line_duration_days']
                available_cols = [col for col in cohort_cols if col in cohort_df.columns]
                cohort_info = cohort_df[['stay_id'] + available_cols]
                combined_features = combined_features.merge(cohort_info, on='stay_id', how='left')
                print(f"Final feature set: {len(combined_features)} rows")
                
            return combined_features
                
        except Exception as e:
            print(f"Error in get_batch_features: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _combine_features(self, features):
        try:
            combined = features['demographics'].copy()
            for key in ['devices', 'labs', 'medications', 'clinical']:
                if features.get(key) is not None and not features[key].empty:
                    combined = combined.merge(features[key], on='stay_id', how='left')
            severity = features.get('severity', {})
            if severity.get('sofa') is not None and not severity['sofa'].empty:
                combined = combined.merge(severity['sofa'], on='stay_id', how='left')
            if severity.get('apsiii') is not None and not severity['apsiii'].empty:
                apsiii_cols = ['stay_id', 'apsiii', 'apsiii_prob']
                combined = combined.merge(severity['apsiii'][apsiii_cols], on='stay_id', how='left')
            if severity.get('sapsii') is not None and not severity['sapsii'].empty:
                sapsii_cols = ['stay_id', 'sapsii', 'sapsii_prob']
                combined = combined.merge(severity['sapsii'][sapsii_cols], on='stay_id', how='left')
            if features.get('chg_bath') is not None and not features['chg_bath'].empty:
                chg_cols = [col for col in features['chg_bath'].columns if col != 'stay_id']
                combined = combined.merge(features['chg_bath'][['stay_id'] + chg_cols], on='stay_id', how='left')
            if features.get('line_care') is not None and not features['line_care'].empty:
                line_features = features['line_care']
                numeric_cols = line_features.select_dtypes(include=['float64', 'int64']).columns.difference(['stay_id'])
                categorical_cols = line_features.select_dtypes(include=['object']).columns.difference(['stay_id'])
                
                aggs = {}
                for col in numeric_cols:
                    aggs[f"{col}_min"] = (col, 'min')
                    aggs[f"{col}_max"] = (col, 'max')
                    aggs[f"{col}_mean"] = (col, 'mean')
                for col in categorical_cols:
                    aggs[f"{col}_types"] = (col, lambda x: '|'.join(sorted(x.dropna().unique())))
                
                line_care_agg = line_features.groupby('stay_id').agg(**aggs).reset_index()
                combined = combined.merge(line_care_agg, on='stay_id', how='left')
            
            # Safely create a simplified line type column
            if 'line_type' in combined.columns:
                def simplify_line_type(x):
                    if isinstance(x, str):
                        if 'Multi Lumen' in x:
                            return 'Multi-Lumen'
                        elif 'Dialysis' in x:
                            return 'Dialysis'
                        elif 'PICC' in x:
                            return 'PICC'
                    return 'Other'
                combined['line_type_simple'] = combined['line_type'].apply(simplify_line_type)
            
            numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns.difference(['stay_id'])
            for col in numeric_cols:
                combined[col].fillna(combined[col].median(), inplace=True)
            categorical_cols = combined.select_dtypes(include=['object']).columns.difference(['stay_id'])
            for col in categorical_cols:
                if combined[col].isnull().any():
                    mode_value = combined[col].mode().iloc[0] if not combined[col].mode().empty else 'Unknown'
                    combined[col].fillna(mode_value, inplace=True)
            
            return combined
            
        except Exception as e:
            import traceback
            print(f"Error combining features: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_feature_names(self):
        example_features = self.get_all_features(self.example_stay_id)
        if example_features is not None:
            return example_features.columns.tolist()
        return None
        
    def get_feature_stats(self, features_df):
        try:
            stats = {}
            numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            stats['numeric'] = features_df[numeric_cols].describe()
            categorical_cols = features_df.select_dtypes(include=['object']).columns
            stats['categorical'] = {
                col: features_df[col].value_counts() for col in categorical_cols
            }
            stats['missing'] = features_df.isnull().sum()
            return stats
        except Exception as e:
            print(f"Error calculating feature statistics: {str(e)}")
            return None


# Testing routines below (if run as a script)
if __name__ == "__main__":
    engine = connect_db()
    
    print("Loading CLABSI cohort...")
    clabsi_cohort = get_clabsi_cohort(engine)
    print("CLABSI cohort loaded successfully!")
    
    if not clabsi_cohort.empty:
        control_cohort = get_control_cohort(engine, clabsi_cohort)
    
    if not clabsi_cohort.empty:
        print("Testing demographic feature extraction...")
        try:
            demographic_features = get_demographic_features(engine, clabsi_cohort)
            print("\nShape of extracted features:", demographic_features.shape)
            print("\nFeature names:")
            print(demographic_features.columns.tolist())
            print("\nMissing values summary:")
            print(demographic_features.isnull().sum()[demographic_features.isnull().sum() > 0])
        except Exception as e:
            print("Error during feature extraction:", str(e))
    else:
        print("Please run the CLABSI cohort creation first")
    
    if not clabsi_cohort.empty:
        print("Testing device feature extraction...")
        try:
            device_features = get_device_features(engine, clabsi_cohort)
            print("\nShape of extracted features:", device_features.shape)
            print("\nFeature names:")
            print(device_features.columns.tolist())
            print("\nMissing values summary:")
            print(device_features.isnull().sum()[device_features.isnull().sum() > 0])
        except Exception as e:
            print("Error during device feature extraction:", str(e))
    else:
        print("Please run the CLABSI cohort creation first")
    
    if not clabsi_cohort.empty:
        print("Testing laboratory value extraction...")
        try:
            lab_features = get_lab_values(engine, clabsi_cohort)
            print("\nShape of extracted features:", lab_features.shape)
            print("\nFirst few rows of data:")
            print(lab_features.head())
        except Exception as e:
            print("Error during feature extraction:", str(e))
    else:
        print("Please run the CLABSI cohort creation first")
    
    severity_scores = get_all_severity_scores(engine, clabsi_cohort)
    if severity_scores['sofa'] is not None:
        print("\nFirst-Day SOFA Scores:")
        print(severity_scores['sofa'].head())
    if severity_scores['apsiii'] is not None:
        print("\nAPS-III Scores:")
        print(severity_scores['apsiii'].head())
    if severity_scores['sapsii'] is not None:
        print("\nSAPS-II Scores:")
        print(severity_scores['sapsii'].head())
    
    line_care_df = get_line_care(engine, clabsi_cohort)
    print("\nSample Central Line Care Metrics:")
    print(line_care_df.head())
    
    thresholds = [3, 5, 7, 10]
    threshold_summary = threshold_analysis(clabsi_cohort, thresholds)
    print("\nThreshold Analysis Summary:")
    print(threshold_summary)
    
    chg_bath_features = get_chg_bath_features(engine, clabsi_cohort)
    print("\nCHG Bath Features:")
    print(chg_bath_features.head())
    
    try:
        medication_features = get_medications(engine, clabsi_cohort)
    except Exception as e:
        print(f"Error executing get_medications: {e}")
    
    clinical_features = get_clinical_features(engine, clabsi_cohort)
    print("\nClinical Features:")
    print(clinical_features.head())
    
    site_risk_df = analyze_line_sites_clabsi_risk(engine, clabsi_cohort, line_care_df)
    
    if engine:
        feature_processor = CLABSIFeatureProcessor(engine)
        print("Testing feature extraction with CLABSI case...")
        test_stay_id = clabsi_cohort['stay_id'].iloc[0]
        features = feature_processor.get_all_features(test_stay_id)
        if features is not None:
            print(f"\nExtracted {len(features.columns)} features for stay_id {test_stay_id}")
            print("\nFeature sample:")
            print(features.head())
            stats = feature_processor.get_feature_stats(features)
            if stats:
                print("\nFeature Statistics (Numeric):")
                print(stats['numeric'])
