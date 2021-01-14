-- MUST BE RUN AFTER labels.sql

-- creates a materialized view flat which looks like this:
/*
 patientunitstayid | gender | age |    ethnicity     | admissionheight | admissionweight | unitadmittime24 |   unittype   |   unitadmitsource    | unitstaytype |     physicianspeciality      | intubated | vent | dialysis | eyes | motor | verbal | meds
-------------------+--------+-----+------------------+-----------------+-----------------+-----------------+--------------+----------------------+--------------+------------------------------+-----------+------+----------+------+-------+--------+------
            141168 | Female | 70  | Caucasian        |          152.40 |           84.30 | 15:54:00        | Med-Surg ICU | Direct Admit         | admit        | critical care medicine (CCM) |         0 |    0 |        0 |    4 |     6 |      5 |    0
            141194 | Male   | 68  | Caucasian        |          180.30 |           73.90 | 07:18:00        | CTICU        | Floor                | admit        | critical care medicine (CCM) |         0 |    0 |        0 |    3 |     6 |      4 |    0
            141203 | Female | 77  | Caucasian        |          160.00 |           70.20 | 20:39:00        | Med-Surg ICU | Floor                | admit        | hospitalist                  |         0 |    1 |        0 |    1 |     3 |      1 |    0
*/

-- delete the materialized view flat if it already exists
drop materialized view if exists flat cascade;
create materialized view flat as
  -- for some reason lots of multiple records are produced, the distinct gets rid of these
  select distinct la.patientunitstayid, p.gender, p.age, p.ethnicity, p.admissionheight, p.admissionweight,
    p.apacheadmissiondx, extract(hour from to_timestamp(p.unitadmittime24,'HH24:MI:SS')) as hour, p.unittype,
    p.unitadmitsource, p.unitstaytype, apr.physicianspeciality, aps.intubated, aps.vent, aps.dialysis, aps.eyes,
    aps.motor, aps.verbal, aps.meds
    from patient as p
    inner join apacheapsvar as aps on aps.patientunitstayid = p.patientunitstayid
    inner join apachepatientresult as apr on apr.patientunitstayid = p.patientunitstayid
    -- to narrow down to the cohort we want to use. We get 89143 stays.
    inner join labels as la on la.patientunitstayid = p.patientunitstayid;
