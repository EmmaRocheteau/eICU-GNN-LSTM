-- creates all the tables and produces csv files
-- takes a few minutes to run

\i /Users/emmarocheteau/PycharmProjects/catherine/eICU_preprocessing/labels.sql
\i /Users/emmarocheteau/PycharmProjects/catherine/eICU_preprocessing/diagnoses.sql
\i /Users/emmarocheteau/PycharmProjects/catherine/eICU_preprocessing/flat_features.sql
\i /Users/emmarocheteau/PycharmProjects/catherine/eICU_preprocessing/timeseries.sql

-- we need to make sure that we have at least some form of time series for every patient in diagnoses, flat and labels
drop materialized view if exists timeseries_patients cascade;
create materialized view timeseries_patients as
  with repeats as (
    select distinct patientunitstayid
      from timeserieslab
    union
    select distinct patientunitstayid
      from timeseriesresp
    union
    select distinct patientunitstayid
      from timeseriesperiodic
    union
    select distinct patientunitstayid
      from timeseriesaperiodic)
  select distinct patientunitstayid
    from repeats;

\copy (select * from labels as l where l.patientunitstayid in (select * from timeseries_patients)) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/labels.csv' with csv header
\copy (select * from diagnoses as d where d.patientunitstayid in (select * from timeseries_patients)) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/diagnoses.csv' with csv header
\copy (select * from flat as f where f.patientunitstayid in (select * from timeseries_patients)) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/flat_features.csv' with csv header
\copy (select * from timeserieslab) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/timeserieslab.csv' with csv header
\copy (select * from timeseriesresp) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/timeseriesresp.csv' with csv header
\copy (select * from timeseriesperiodic) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/timeseriesperiodic.csv' with csv header
\copy (select * from timeseriesaperiodic) to '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/timeseriesaperiodic.csv' with csv header