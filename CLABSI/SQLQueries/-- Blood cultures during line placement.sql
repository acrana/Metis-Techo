-- Blood cultures during line placement
WITH line_times AS (
    SELECT subject_id, starttime, endtime
    FROM mimiciv_icu.procedureevents
    WHERE ordercategoryname = 'Invasive Lines'
)
SELECT 
    lt.subject_id,
    lt.starttime as line_start,
    lt.endtime as line_end,
    m.charttime as culture_time,
    m.spec_type_desc,
    m.org_name,
    m.interpretation
FROM line_times lt
LEFT JOIN mimiciv_hosp.microbiologyevents m 
    ON lt.subject_id = m.subject_id
    AND m.charttime BETWEEN lt.starttime AND lt.endtime
WHERE LOWER(m.spec_type_desc) LIKE '%blood%';