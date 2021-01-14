import pandas as pd


def preprocess_flat(flat):

    # make naming consistent with the other tables
    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)

    # admission diagnosis is dealt with in diagnoses.py not flat features
    flat.drop(columns=['apacheadmissiondx'], inplace=True)

    flat['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=['ethnicity', 'unittype', 'unitadmitsource', 'unitstaytype',
                                         'physicianspeciality'])

    # 2 out of 89123 patients have NaN for age; we fill this with the mean value which is 64
    flat['age'].fillna('64', inplace=True)
    # some of the ages are like '> 89' rather than numbers, this needs removing and converting to numbers
    # but we make an extra variable to keep this information
    flat['> 89'] = flat['age'].str.contains('> 89').astype(int)
    flat['age'] = flat['age'].replace('> ', '', regex=True)
    flat['age'] = [float(value) for value in flat.age.values]

    # note that the features imported from the time series have already been normalised
    # standardisation is for features that are probably normally distributed
    features_for_standardisation = 'admissionheight'
    means = flat[features_for_standardisation].mean(axis=0)
    stds = flat[features_for_standardisation].std(axis=0)
    # standardise
    flat[features_for_standardisation] = (flat[features_for_standardisation] - means) / stds

    # probably not normally distributed
    features_for_min_max = ['admissionweight', 'age', 'eyes', 'motor', 'verbal', 'hour']
    # minus the minimum value and then divide by the maximum value
    flat[features_for_min_max] -= flat[features_for_min_max].min()
    flat[features_for_min_max] /= flat[features_for_min_max].max()

    # preen the features by removing any really uncommon ones maybe - or coalesce some
    return flat

def preprocess_labels(labels):

    # make naming consistent with the other tables
    labels.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    labels.set_index('patient', inplace=True)

    labels['actualhospitalmortality'].replace({'EXPIRED': 1, 'ALIVE': 0}, inplace=True)

    return labels

def flat_and_labels_main(eICU_path):

    print('==> Loading data from labels and flat features files...')
    flat = pd.read_csv(eICU_path + 'flat_features.csv')
    flat = preprocess_flat(flat)
    labels = pd.read_csv(eICU_path + 'labels.csv')
    labels = preprocess_labels(labels)

    print('==> Saving finalised preprocessed labels and flat features...')
    flat.to_csv(eICU_path + 'preprocessed_flat.csv')
    labels.to_csv(eICU_path + 'preprocessed_labels.csv')
    return

if __name__=='__main__':
    eICU_path = '/Users/emmarocheteau/PycharmProjects/catherine/eICU_data/'
    flat_and_labels_main(eICU_path)