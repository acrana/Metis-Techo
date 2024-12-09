-- Track antibiotic use during line placement
SELECT 
    p.subject_id,
    pe.starttime as line_start,
    pe.endtime as line_end,
    ph.starttime as med_start,
    ph.stoptime as med_stop,
    ph.drug
FROM mimiciv_icu.procedureevents pe
INNER JOIN mimiciv_hosp.patients p ON pe.subject_id = p.subject_id
INNER JOIN mimiciv_hosp.prescriptions ph ON p.subject_id = ph.subject_id
WHERE pe.ordercategoryname = 'Invasive Lines'
AND ph.starttime BETWEEN pe.starttime AND pe.endtime;