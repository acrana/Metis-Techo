-- Get mortality and length of stay
SELECT 
    p.subject_id,
    a.hospital_expire_flag,
    a.admittime,
    a.dischtime,
    EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400.0 as los_days
FROM mimiciv_hosp.patients p
INNER JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
WHERE p.subject_id IN (
    SELECT DISTINCT subject_id 
    FROM mimiciv_icu.procedureevents
    WHERE ordercategoryname = 'Invasive Lines'
);