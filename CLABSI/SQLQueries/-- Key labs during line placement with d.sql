-- Key labs during line placement with descriptions
WITH line_patients AS (
    SELECT subject_id, starttime, endtime
    FROM mimiciv_icu.procedureevents
    WHERE ordercategoryname = 'Invasive Lines'
)
SELECT 
    lp.subject_id,
    lp.starttime as line_start,
    lp.endtime as line_end,
    le.charttime,
    d.label as lab_name,
    d.fluid as specimen_type,
    d.category as lab_category,
    le.valuenum,
    le.valueuom
FROM line_patients lp
INNER JOIN mimiciv_hosp.labevents le 
    ON lp.subject_id = le.subject_id
INNER JOIN mimiciv_hosp.d_labitems d 
    ON le.itemid = d.itemid
WHERE le.charttime BETWEEN lp.starttime AND lp.endtime
AND d.category IN (
    'Blood Gas',
    'Chemistry',
    'Hematology',
    'Coagulation'
)
ORDER BY lp.subject_id, le.charttime;