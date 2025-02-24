-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS mimiciv_derived.first_day_bg; CREATE TABLE mimiciv_derived.first_day_bg AS
/* Highest/lowest blood gas values for all blood specimens, */ /* including venous/arterial/mixed */
SELECT
  ie.subject_id,
  ie.stay_id,
  MIN(lactate) AS lactate_min,
  MAX(lactate) AS lactate_max,
  MIN(ph) AS ph_min,
  MAX(ph) AS ph_max,
  MIN(so2) AS so2_min,
  MAX(so2) AS so2_max,
  MIN(po2) AS po2_min,
  MAX(po2) AS po2_max,
  MIN(pco2) AS pco2_min,
  MAX(pco2) AS pco2_max,
  MIN(aado2) AS aado2_min,
  MAX(aado2) AS aado2_max,
  MIN(aado2_calc) AS aado2_calc_min,
  MAX(aado2_calc) AS aado2_calc_max,
  MIN(pao2fio2ratio) AS pao2fio2ratio_min,
  MAX(pao2fio2ratio) AS pao2fio2ratio_max,
  MIN(baseexcess) AS baseexcess_min,
  MAX(baseexcess) AS baseexcess_max,
  MIN(bicarbonate) AS bicarbonate_min,
  MAX(bicarbonate) AS bicarbonate_max,
  MIN(totalco2) AS totalco2_min,
  MAX(totalco2) AS totalco2_max,
  MIN(hematocrit) AS hematocrit_min,
  MAX(hematocrit) AS hematocrit_max,
  MIN(hemoglobin) AS hemoglobin_min,
  MAX(hemoglobin) AS hemoglobin_max,
  MIN(carboxyhemoglobin) AS carboxyhemoglobin_min,
  MAX(carboxyhemoglobin) AS carboxyhemoglobin_max,
  MIN(methemoglobin) AS methemoglobin_min,
  MAX(methemoglobin) AS methemoglobin_max,
  MIN(temperature) AS temperature_min,
  MAX(temperature) AS temperature_max,
  MIN(chloride) AS chloride_min,
  MAX(chloride) AS chloride_max,
  MIN(calcium) AS calcium_min,
  MAX(calcium) AS calcium_max,
  MIN(glucose) AS glucose_min,
  MAX(glucose) AS glucose_max,
  MIN(potassium) AS potassium_min,
  MAX(potassium) AS potassium_max,
  MIN(sodium) AS sodium_min,
  MAX(sodium) AS sodium_max
FROM mimiciv_icu.icustays AS ie
LEFT JOIN mimiciv_derived.bg AS bg
  ON ie.subject_id = bg.subject_id
  AND bg.charttime >= ie.intime - INTERVAL '6 HOUR'
  AND bg.charttime <= ie.intime + INTERVAL '1 DAY'
GROUP BY
  ie.subject_id,
  ie.stay_id