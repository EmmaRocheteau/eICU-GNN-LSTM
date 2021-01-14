import pandas as pd
import numpy as np

def add_codes(splits, codes_dict, words_dict, count):
    codes = list()
    levels = len(splits)  # the max number of levels is 6
    if levels >= 1:
        try:
            codes.append(codes_dict[splits[0]][0])
            codes_dict[splits[0]][2] += 1
        except KeyError:
            codes_dict[splits[0]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0]
            count += 1
    if levels >= 2:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][0])
            codes_dict[splits[0]][1][splits[1]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1]
            count += 1
    if levels >= 3:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2]
            count += 1
    if levels >= 4:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3]
            count += 1
    if levels >= 5:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3] + '|' + splits[4]
            count += 1
    if levels is 6:
        try:
            codes.append(codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]][0])
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]][2] += 1
        except KeyError:
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1][splits[5]] = [count, {}, 0]
            codes.append(count)
            words_dict[count] = splits[0] + '|' + splits[1] + '|' + splits[2] + '|' + splits[3] + '|' + splits[4] + '|' + splits[5]
            count += 1
    return codes, count


def get_mapping_dict(unique_diagnoses):

    # a lot of the notes strings look the same, so we will not propagate beyond Organ Systems for this:
    main_diagnoses = [a for a in unique_diagnoses if not a.startswith('notes')]
    pasthistory_organsystems = [a for a in unique_diagnoses if a.startswith('notes/Progress Notes/Past History/Organ Systems/')]
    pasthistory_comments = [a for a in unique_diagnoses if a.startswith('notes/Progress Notes/Past History/Past History Obtain Options')]

    # sort into alphabetical order to keep the codes roughly together numerically.
    main_diagnoses.sort()
    pasthistory_organsystems.sort()
    pasthistory_comments.sort()

    mapping_dict = {}
    codes_dict = {}
    words_dict = {}
    count = 0

    for diagnosis in main_diagnoses:
        splits = diagnosis.split('|')
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_organsystems:
        # take out the things that are common to all of these because it creates unnecessary levels
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Organ Systems/', '')
        splits = shortened.split('/')  # note different split to main_diagnoses
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_comments:
        # take out the things that are common to all of these because it creates unnecessary levels
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Past History Obtain Options/', '')
        splits = shortened.split('/')  # note different split to main_diagnoses
        codes, count = add_codes(splits, codes_dict, words_dict, count)
        # add all codes relevant to the diagnosisstring
        mapping_dict[diagnosis] = codes

    return codes_dict, mapping_dict, count, words_dict

# get rid of anything that is a parent to only one child (index 2 is 1)
def find_pointless_codes(diag_dict):
    pointless_codes = []
    for key, value in diag_dict.items():
        # if there is only one child, then the branch is linear and can be condensed
        if value[2] is 1:
            pointless_codes.append(value[0])
        # get rid of repeat copies where the parent and child are the same title
        for next_key, next_value in value[1].items():
            if key.lower() == next_key.lower():
                pointless_codes.append(next_value[0])
        pointless_codes += find_pointless_codes(value[1])
    return pointless_codes

# get rid of any codes that have a frequency of less than 50
def find_rare_codes(cut_off, sparse_df):
    prevalence = sparse_df.sum(axis=0)  # see if you can stop it making pointless extra classes
    rare_codes = prevalence.loc[prevalence <= cut_off].index
    return list(rare_codes)

def add_adm_diag(sparse_df, eICU_path, cut_off):

    print('==> Adding admission diagnoses from flat_features.csv...')
    flat = pd.read_csv(eICU_path + 'flat_features.csv')
    adm_diag = flat[['patientunitstayid', 'apacheadmissiondx']]
    adm_diag.set_index('patientunitstayid', inplace=True)
    adm_diag = pd.get_dummies(adm_diag, columns=['apacheadmissiondx'])
    rare_adm_diag = find_rare_codes(cut_off, adm_diag)
    # it could be worth doing some further grouping on the rare_adm_diagnoses before we throw them away
    # the first word is a good approximation
    groupby_dict = {}
    for diag in adm_diag.columns:
        if diag in rare_adm_diag:
            groupby_dict[diag] = 'groupedapacheadmissiondx_' + diag.split(' ', 1)[0].split('/', 1)[0].split(',', 1)[0][18:]
        else:
            groupby_dict[diag] = diag
    adm_diag = adm_diag.groupby(groupby_dict, axis=1).sum()
    rare_adm_diag = find_rare_codes(cut_off, adm_diag)
    adm_diag.drop(columns=rare_adm_diag, inplace=True)
    all_diag = sparse_df.join(adm_diag, how='outer', on='patientunitstayid')
    return all_diag

def diagnoses_main(eICU_path, cut_off_prevalence):

    print('==> Loading data diagnoses.csv...')
    diagnoses = pd.read_csv(eICU_path + 'diagnoses.csv')
    diagnoses.set_index('patientunitstayid', inplace=True)

    unique_diagnoses = diagnoses.diagnosisstring.unique()
    codes_dict, mapping_dict, count, words_dict = get_mapping_dict(unique_diagnoses)

    patients = diagnoses.index.unique()
    index_to_patients = dict(enumerate(patients))
    patients_to_index = {v: k for k, v in index_to_patients.items()}

    # reconfiguring the diagnosis data into a dictionary
    diagnoses = diagnoses.groupby('patientunitstayid').apply(lambda diag: diag.to_dict(orient='list')['diagnosisstring']).to_dict()
    diagnoses = {patient: [code for diag in list_diag for code in mapping_dict[diag]] for (patient, list_diag) in diagnoses.items()}

    num_patients = len(patients)
    sparse_diagnoses = np.zeros(shape=(num_patients, count))
    for patient, codes in diagnoses.items():
        sparse_diagnoses[patients_to_index[patient], codes] = 1  # N.B. it doesn't matter that codes contains repeats here

    pointless_codes = find_pointless_codes(codes_dict)

    sparse_df = pd.DataFrame(sparse_diagnoses, index=patients, columns=range(count))
    cut_off = round(cut_off_prevalence*num_patients)
    rare_codes = find_rare_codes(cut_off, sparse_df)
    sparse_df.drop(columns=rare_codes + pointless_codes, inplace=True)
    sparse_df.rename(columns=words_dict, inplace=True)
    sparse_df = add_adm_diag(sparse_df, eICU_path, cut_off)
    print('==> Keeping ' + str(sparse_df.shape[1]) + ' diagnoses which have a prevalence of more than ' + str(cut_off_prevalence*100) + '%...')

    # make naming consistent with the other tables
    sparse_df.rename_axis('patient', inplace=True)

    print('==> Saving finalised preprocessed diagnoses...')
    sparse_df.to_csv(eICU_path + 'preprocessed_diagnoses.csv')

    return

if __name__=='__main__':

    eICU_path = '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/'
    cut_off_prevalence = 0.001  # this would be 0.1%
    diagnoses_main(eICU_path, cut_off_prevalence)