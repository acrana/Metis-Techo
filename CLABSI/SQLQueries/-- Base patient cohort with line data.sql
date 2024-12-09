-- Base patient cohort with line data
SELECT 
    p.subject_id,
    p.gender,
    p.anchor_age,
    pe.hadm_id,
    pe.starttime as line_start,
    pe.endtime as line_end,
    CASE 
        WHEN valueuom = 'min' THEN CAST(pe.value AS numeric)/1440
        WHEN valueuom = 'day' THEN CAST(pe.value AS numeric)
    END as line_days
FROM mimiciv_hosp.patients p
INNER JOIN mimiciv_icu.procedureevents pe 
    ON p.subject_id = pe.subject_id
WHERE pe.ordercategoryname = 'Invasive Lines';