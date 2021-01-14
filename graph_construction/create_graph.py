import numpy as np
from scipy import sparse
import pandas as pd
import torch
import argparse
import json
import os


def get_device_and_dtype():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.cuda.sparse.ByteTensor if device.type == 'cuda' else torch.sparse.ByteTensor
    return device, dtype


def get_freqs(train_diagnoses):
    return train_diagnoses.sum()


def score_matrix(diagnoses, freq_adjustment=None, debug=False):
    print('==> Making score matrix')
    diagnoses = np.array(diagnoses).astype(np.uint8)  # keep the memory requirement small!
    device, dtype = get_device_and_dtype()
    diagnoses = torch.tensor(diagnoses,  device=device).type(dtype)
    if debug:
        diagnoses = diagnoses[:1000]
    print('==> Finding common diagnoses')
    if freq_adjustment is not None:
        # take the inverse to reflect the 'rareness' instead of 'commonness'
        freq_adjustment = 1 / freq_adjustment
        # multiply by 1000 so I can still use integers. It is less precise but still works :)
        # the rare diagnoses are upweighted by about a factor of 4 compared to the commonest
        # the addition of 1 is to ensure that all diagnoses shared are counted
        freq_adjustment = torch.tensor(freq_adjustment * 1000, device=device).type(dtype) + 1
        scores = torch.sparse.mm(diagnoses * freq_adjustment.unsqueeze(0), diagnoses.permute(1, 0))
    else:
        scores = torch.sparse.mm(diagnoses, diagnoses.permute(1, 0))  # only compute top part
    return scores


def make_graph_penalise(diagnoses, scores, batch_size=1000, debug=True, k=3, mode='k_closest', save_edge_values=True):
    print('==> Getting edges')
    if debug:
        diagnoses = diagnoses[:1000]
    no_pts = len(diagnoses)
    diags_per_pt = diagnoses.sum(axis=1)
    diags_per_pt = torch.tensor(diags_per_pt.values).type(torch.ShortTensor)
    del diagnoses

    if save_edge_values:
        edges_val = sparse.lil_matrix((no_pts, no_pts), dtype=np.int16)
    edges = sparse.lil_matrix((no_pts, no_pts), dtype=np.uint8)

    down = torch.split(diags_per_pt.repeat(no_pts, 1), batch_size, dim=0)
    across = torch.split(diags_per_pt.repeat(no_pts, 1).permute(1, 0), batch_size, dim=0)
    scores = scores.fill_diagonal_(0)  # remove self scores on diagonal
    score = torch.split(scores, batch_size, dim=0)
    prev_pts = 0
    for i, (d, a, s) in enumerate(zip(down, across, score)):
        print('==> Processed {} patients'.format(prev_pts))
        total_combined_diags = d + a
        s_pen = 5 * s - total_combined_diags  # the 5 is fairly arbitrary but I don't want to penalise not sharing diagnoses too much
        if mode == 'k_closest':
            k_ = k
        else:
            k_ = 1 # make sure there is at least one edge for each node in the threshold graph
        for patient in range(len(d)):
            k_highest_inds = torch.sort(s_pen[patient].flatten()).indices[-k_:]
            if save_edge_values:
                k_highest_vals = torch.sort(s_pen[patient].flatten()).values[-k_:]
                for i, val in zip(k_highest_inds, k_highest_vals):
                    if val == 0:  # these get removed if val is 0
                        val = 1
                    edges_val[patient + prev_pts, i] = val
            for i in k_highest_inds:
                edges[patient + prev_pts, i] = 1
        prev_pts += batch_size
        if mode == 'threshold':
            scores_lower = torch.tril(s_pen, diagonal=-1)
            if i == 0:  # define threshold
                desired_no_edges = k * len(s_pen)
                threshold_value = torch.sort(scores_lower.flatten()).values[-desired_no_edges]
            # for batch in batch(no_pts, n=10):
            for batch in torch.split(scores_lower, 100, dim=0):
                batch[batch < threshold_value] = 0
            edges[batch_size * i:batch_size * i + len(scores_lower)] = \
                edges[batch_size * i:batch_size * i + len(scores_lower)] + \
                sparse.lil_matrix(scores_lower)

    del scores, score, down, across, d, a, s, total_combined_diags, s_pen

    # make it symmetric again
    edges = edges + edges.transpose()
    if save_edge_values:
        edges_val = edges_val + edges_val.transpose()
        for i, (edge, edge_val) in enumerate(zip(edges, edges_val)):
            edges_val[i, edge.indices] = edge_val.data // edge.data
        edges = edges_val
    edges.setdiag(0)  # remove any left over self edges from patients without any diagnoses (these will be generally matched with others having no diagnoses)
    edges.eliminate_zeros()
    # do upper triangle again and then save
    edges = sparse.tril(edges, k=-1)
    v, u, vals = sparse.find(edges)
    return u, v, vals, k


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx, min(ndx + n, l)]


def make_graph(scores, threshold=True, k_closest=False, k=3):
    print('==> Getting edges')
    no_pts = len(scores)
    if k_closest:
        k_ = k
    else:
        k_ = 1 # ensure there is at least one edge per node in the threshold graph
    edges = sparse.lil_matrix((no_pts, no_pts), dtype=np.uint8)
    scores.fill_diagonal_(0)  # get rid of self connection scores
    for patient in range(no_pts):
        k_highest = torch.sort(scores[patient].flatten()).indices[-k_:]
        for i in k_highest:
            edges[patient, i] = 1
    del scores
    edges = edges + edges.transpose()  # make it symmetric again
    # do upper triangle again and then save
    edges = sparse.tril(edges, k=-1)
    if threshold:
        scores_lower = torch.tril(scores, diagonal=-1)
        del scores
        desired_no_edges = k * no_pts
        threshold_value = torch.sort(scores_lower.flatten()).values[-desired_no_edges]
        #for batch in batch(no_pts, n=10):
        for batch in torch.split(scores_lower, 100, dim=0):
            batch[batch < threshold_value] = 0
        edges = edges + sparse.lil_matrix(scores_lower)
        del scores_lower
    v, u, _ = sparse.find(edges)
    return u, v, k


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--mode', type=str, default='k_closest', help='k_closest or threshold')
    parser.add_argument('--freq_adjust', action='store_true')
    parser.add_argument('--penalise_non_shared', action='store_true')
    args = parser.parse_args()

    print(args)

    with open('paths.json', 'r') as f:
        eICU_path = json.load(f)["eICU_path"]
        graph_dir = json.load(f)["graph_dir"]

    device, dtype = get_device_and_dtype()
    adjust = '_adjusted' if args.freq_adjust else ''

    if not os.path.exists(graph_dir):  # make sure the graphs folder exists
        os.makedirs(graph_dir)

    # get score matrix
    try:
        scores = torch.load('{}scores{}.pt'.format(graph_dir, adjust))
        print('==> Loaded existing scores matrix')
    except FileNotFoundError:
        train_diagnoses = pd.read_csv('{}train/diagnoses.csv'.format(eICU_path), index_col='patient')
        val_diagnoses = pd.read_csv('{}val/diagnoses.csv'.format(eICU_path), index_col='patient')
        test_diagnoses = pd.read_csv('{}test/diagnoses.csv'.format(eICU_path), index_col='patient')
        all_diagnoses = pd.concat([train_diagnoses, val_diagnoses, test_diagnoses], sort=False)
        if args.freq_adjust:
            freq_adjustment = get_freqs(train_diagnoses)
        else:
            freq_adjustment = None
        del train_diagnoses, val_diagnoses, test_diagnoses
        scores = score_matrix(all_diagnoses, freq_adjustment=freq_adjustment)
        del all_diagnoses
        torch.save(scores, '{}scores{}.pt'.format(graph_dir, adjust))

    # make graph
    if args.penalise_non_shared:
        adjust = '_adjusted_ns'
        train_diagnoses = pd.read_csv('{}train/diagnoses.csv'.format(eICU_path), index_col='patient')
        val_diagnoses = pd.read_csv('{}val/diagnoses.csv'.format(eICU_path), index_col='patient')
        test_diagnoses = pd.read_csv('{}test/diagnoses.csv'.format(eICU_path), index_col='patient')
        all_diagnoses = pd.concat([train_diagnoses, val_diagnoses, test_diagnoses], sort=False)
        del train_diagnoses, val_diagnoses, test_diagnoses
        u, v, vals, k = make_graph_penalise(all_diagnoses, scores, k=args.k)
    else:
        if args.mode == 'threshold':
            u, v, k = make_graph(scores, threshold=True, k_closest=False, k=args.k)
        else:
            u, v, k = make_graph(scores, threshold=False, k_closest=True, k=args.k)
    np.savetxt('{}{}_u_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), u.astype(int), fmt='%i')
    np.savetxt('{}{}_v_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), v.astype(int), fmt='%i')
    np.savetxt('{}{}_scores_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), vals.astype(int), fmt='%i')