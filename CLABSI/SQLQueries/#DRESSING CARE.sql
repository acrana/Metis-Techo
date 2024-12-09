#DRESSING CARE
WITH line_times AS (
    SELECT subject_id, starttime, endtime
    FROM mimiciv_icu.procedureevents
    WHERE ordercategoryname = 'Invasive Lines'
)
SELECT 
    lt.subject_id,
    lt.starttime as line_start,
    lt.endtime as line_end,
    c.charttime,
    di.itemid,
    di.label as dressing_type,
    c.value as documentation
FROM line_times lt
INNER JOIN mimiciv_icu.chartevents c 
    ON lt.subject_id = c.subject_id
INNER JOIN mimiciv_icu.d_items di 
    ON c.itemid = di.itemid
WHERE c.charttime BETWEEN lt.starttime AND lt.endtime
AND (
    LOWER(di.label) LIKE '%dressing%' 
    OR LOWER(di.label) LIKE '%sterile%'
    OR LOWER(di.label) LIKE '%line care%'
)
ORDER BY lt.subject_id, c.charttime;