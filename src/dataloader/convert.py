import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import write_json, write_pkl, load_json


def convert_timeseries_into_mmap(data_dir, save_dir, n_rows=100000):
    """
    read csv file and convert time series data into mmap file.
    """
    save_path = Path(save_dir) / 'ts.dat'
    shape = (n_rows, 24, 34)
    write_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)
    ids = []
    n = 0
    info = {}
    info['name'] = 'ts'
    
    for split in ['train', 'val', 'test']:
        print('split: ', split)
        csv_path = Path(data_dir) / split / 'timeseries.csv'
        df = pd.read_csv(csv_path)
        arr = df.values
        new = np.reshape(arr, (-1, 24, 35))
        pos_to_id = new[:, 0, 0]
        ids.append(pos_to_id)
        new = new[:, :, 1:] # no patient column
        write_file[n : n+len(new), :, :] = new
        info[split + '_len'] = len(new)
        n += len(new)
        del new, arr
    
    info['total'] = n
    info['shape'] = shape
    info['columns'] = list(df)[1:]
    del df

    ids = np.concatenate(ids)
    id2pos = {pid: pos for pos, pid in enumerate(ids)}
    pos2id = {pos:pid for pos, pid in enumerate(ids)}
    
    assert len(set(ids)) == len(ids)

    print('saving..')
    write_pkl(id2pos, Path(save_dir) / 'id2pos.pkl')
    write_pkl(pos2id, Path(save_dir) / 'pos2id.pkl')
    write_json(info, Path(save_dir) / 'ts_info.json')
    print(info)


def convert_into_mmap(data_dir, save_dir, csv_name, n_cols=None, n_rows=100000):
    """
    read csv file and convert flat data into mmap file.
    """
    csv_to_cols = {'diagnoses': 520, 'diagnoses_1033': 1034, 'labels': 5, 'flat': 58} # including patient column
    n_cols = (csv_to_cols[csv_name] -1) if n_cols is None else n_cols
    shape = (n_rows, n_cols)

    save_path = Path(save_dir) / f'{csv_name}.dat'
    write_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)

    info = {'name': csv_name, 'shape': shape}

    n = 0

    for split in ['train', 'val', 'test']:
        print('split: ', split)
        csv_path = Path(data_dir) / split / f'{csv_name}.csv'
        df = pd.read_csv(csv_path)
        arr = df.values[:, 1:] # cut out patient column
        arr_len = len(arr)
        write_file[n : n+arr_len, :] = arr # write into mmap
        info[split + '_len'] = arr_len
        n += arr_len
        del arr
    
    info['total'] = n
    info['columns'] = list(df)[1:]

    write_json(info, Path(save_dir) / f'{csv_name}_info.json')
    print(info)
    

def read_mm(datadir, name):
    """
    name can be one of {ts, diagnoses, labels, flat}.
    """
    info = load_json(Path(datadir) / (name + '_info.json'))
    dat_path = Path(datadir) / (name + '.dat')
    data = np.memmap(dat_path, dtype=np.float32, shape=tuple(info['shape']))
    return data, info


if __name__ == '__main__':
    paths = load_json('paths.json')
    data_dir = paths['eICU_path']
    save_dir = paths['data_dir']
    print(f'Load eICU processed data from {data_dir}')
    print(f'Saving mmap data in {save_dir}')
    print('--'*30)
    Path(save_dir).mkdir(exist_ok=True)
    print('** Converting time series **')
    convert_timeseries_into_mmap(data_dir, save_dir)
    for csv_name in ['flat', 'diagnoses', 'labels']:
        print(f'** Converting {csv_name} **')
        convert_into_mmap(data_dir, save_dir, csv_name)
    print('--'*30)
    print(f'Done! Saved data in {save_dir}')