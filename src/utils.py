import os
import csv
import pickle
import json


def write_pkl(data, path, verbose=1):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('saved to ', path)

def write_json(data, path, sort_keys=False, verbose=1):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=4)
    if verbose:
        print('saved to ', path)


def load_json(path):
    with open(path, 'r') as json_file:
        info = json.load(json_file)
    return info

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def record_results(path_results, hparams, log_dict):
    results = {r: log_dict[r] for r in log_dict if r != 'test_conf_m'}
    header = [h for h in hparams] + [r for r in results]
    ret = {**hparams, **results}
    file_exists = os.path.isfile(path_results)
    with open(path_results, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(ret)
    print('Written results at ', path_results)
