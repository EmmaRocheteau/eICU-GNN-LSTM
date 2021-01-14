import pandas as pd
import json

with open('paths.json', 'r') as f:
    eICU_path = json.load(f)["eICU_path"]

train_diagnoses = pd.read_csv('{}train/diagnoses.csv'.format(eICU_path), index_col='patient')
val_diagnoses = pd.read_csv('{}val/diagnoses.csv'.format(eICU_path), index_col='patient')
test_diagnoses = pd.read_csv('{}test/diagnoses.csv'.format(eICU_path), index_col='patient')

diag_strings = train_diagnoses.columns

# some quick cleaning i.e. remove classes and subclasses, get rid of strange words and characters
cleaned_strings = []
for i, diag in enumerate(diag_strings):
    diag = diag.replace('groupedapacheadmissiondx', '').replace('apacheadmissiondx', '').replace(' (R)', '')  # get rid of non-useful text
    cleaned_strings.append(diag)
# hacky way to quickly get indexable list
train_diagnoses.columns = cleaned_strings
cleaned_strings = train_diagnoses.columns

def get_diagnosis_strings(diagnoses_df, partition=''):
    with open(eICU_path + partition + '/diagnosis_strings_cleaned.txt', 'w') as f:
        for i, row in diagnoses_df.iterrows():
            diagnosis_strings = cleaned_strings[row.to_numpy().nonzero()[0]]
            patient_strings = []
            for diag in diagnosis_strings:
                if not any(diag in string for string in diagnosis_strings.drop(diag)):  # check in the rest of the strings for overlap
                    patient_strings.append(diag.replace('_', ' ').replace('|', ' '))
            str_to_write = ", ".join(patient_strings)
            if str_to_write == "":
                str_to_write = "No Diagnoses"
            f.write(str_to_write)
            f.write("\n")

get_diagnosis_strings(train_diagnoses, partition='train')
get_diagnosis_strings(val_diagnoses, partition='val')
get_diagnosis_strings(test_diagnoses, partition='test')