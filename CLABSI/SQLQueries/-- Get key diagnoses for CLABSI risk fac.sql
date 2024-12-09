-- Get key diagnoses for CLABSI risk factors
WITH line_patients AS (
    SELECT DISTINCT subject_id 
    FROM mimiciv_icu.procedureevents
    WHERE ordercategoryname = 'Invasive Lines'
),
categorized_conditions AS (
    SELECT 
        d.subject_id,
        d.icd_code,
        d.icd_version,
        CASE
            -- Neutropenia
            WHEN d.icd_code LIKE '288.0%' OR d.icd_code LIKE 'D70%' THEN 'Neutropenia'
            
            -- Cancer/Malignancy
            WHEN d.icd_code LIKE '14%' OR d.icd_code LIKE '15%' OR d.icd_code LIKE '16%' OR d.icd_code LIKE '17%' OR 
                 d.icd_code LIKE '18%' OR d.icd_code LIKE '19%' OR d.icd_code LIKE '20%' OR d.icd_code LIKE '21%' OR
                 d.icd_code LIKE 'C%' THEN 'Cancer'
            
            -- Diabetes
            WHEN d.icd_code LIKE '250%' OR d.icd_code LIKE 'E11%' OR d.icd_code LIKE 'E10%' THEN 'Diabetes'
            
            -- Renal Failure
            WHEN d.icd_code LIKE '584%' OR d.icd_code LIKE '585%' OR d.icd_code LIKE '586%' OR 
                 d.icd_code LIKE 'N17%' OR d.icd_code LIKE 'N18%' OR d.icd_code LIKE 'N19%' THEN 'Renal Failure'
            
            -- Sepsis
            WHEN d.icd_code LIKE '995.91%' OR d.icd_code LIKE '995.92%' OR d.icd_code LIKE 'A41%' OR 
                 d.icd_code LIKE 'R65.2%' THEN 'Sepsis'
            
            -- Immunocompromised
            WHEN d.icd_code LIKE '279%' OR d.icd_code LIKE 'D80%' OR d.icd_code LIKE 'D81%' OR 
                 d.icd_code LIKE 'D82%' OR d.icd_code LIKE 'D83%' OR d.icd_code LIKE 'D84%' THEN 'Immunocompromised'
            
            -- Burns
            WHEN d.icd_code LIKE '940%' OR d.icd_code LIKE '941%' OR d.icd_code LIKE '942%' OR d.icd_code LIKE '943%' OR 
                 d.icd_code LIKE '944%' OR d.icd_code LIKE '945%' OR d.icd_code LIKE '946%' OR d.icd_code LIKE '947%' OR 
                 d.icd_code LIKE '948%' OR d.icd_code LIKE '949%' OR d.icd_code LIKE 'T20%' OR d.icd_code LIKE 'T21%' OR 
                 d.icd_code LIKE 'T22%' OR d.icd_code LIKE 'T23%' OR d.icd_code LIKE 'T24%' OR d.icd_code LIKE 'T25%' OR 
                 d.icd_code LIKE 'T26%' OR d.icd_code LIKE 'T27%' OR d.icd_code LIKE 'T28%' OR d.icd_code LIKE 'T29%' OR 
                 d.icd_code LIKE 'T30%' OR d.icd_code LIKE 'T31%' OR d.icd_code LIKE 'T32%' THEN 'Burns'
        END as condition_category
    FROM line_patients lp
    INNER JOIN mimiciv_hosp.diagnoses_icd d ON lp.subject_id = d.subject_id
)
SELECT * FROM categorized_conditions 
WHERE condition_category IS NOT NULL
ORDER BY subject_id, condition_category;