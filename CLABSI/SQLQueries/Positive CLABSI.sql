SELECT DISTINCT 
    p.subject_id,
    p.gender,
    p.anchor_age as age,
    d.icd_code,
    di.long_title as diagnosis
FROM mimiciv_hosp.diagnoses_icd d
INNER JOIN mimiciv_hosp.patients p 
    ON d.subject_id = p.subject_id
INNER JOIN mimiciv_hosp.d_icd_diagnoses di 
    ON d.icd_code = di.icd_code
WHERE d.icd_code IN ('99931', '99932', 'T80.2')  -- Specific CLABSI codes
ORDER BY p.subject_id;