import numpy as np
from scipy import sparse
import torch
import argparse
from graph_construction.create_graph import get_device_and_dtype
import json


def make_graph_bert(bert, batch_size=1000, debug=False, k=3, mode='k_closest', eps=0.005):
    print('==> Getting edges')
    no_pts = 89123 if debug is False else 10000
    edges = sparse.lil_matrix((no_pts, no_pts), dtype=np.uint8)
    edges_val = sparse.lil_matrix((no_pts, no_pts), dtype=np.float16)
    device, dtype = get_device_and_dtype()
    if debug:
        bert = bert[:10000]
    bert = torch.tensor(bert, device=device).type(dtype)
    for i, batch in enumerate(torch.split(bert, batch_size, dim=0)):
        distances = torch.cdist(batch, bert, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        print('==> Done {} patients'.format((i + 1) * batch_size))
        if mode == 'k_closest':
            k_ = k + 1  # this will select the k + 1 closest (the self edge will be removed)
        else:  # threshold
            k_ = 1  # to ensure that every node has at least one edge for the threshold approach
        for patient in range(len(distances)):
            k_lowest_inds = torch.sort(distances[patient].flatten()).indices.cpu()[:k_]
            k_lowest_vals = torch.sort(distances[patient].flatten()).values.cpu()[:k_]
            for j, val in zip(k_lowest_inds, k_lowest_vals):
                if val == 0:  # these get removed if val is 0
                    val = eps
                edges_val[patient + batch_size * i, j] = val
                edges[patient + batch_size * i, j] = 1
        if mode == 'threshold':
            distances_lower = torch.tril(distances, diagonal=-1)
            if i == 0:  # define threshold
                desired_no_edges = k * len(distances)
                threshold_value = torch.sort(distances_lower.flatten()).values[desired_no_edges]
            # for batch in batch(no_pts, n=10):
            for batch in torch.split(distances_lower, 100, dim=0):
                batch[batch > threshold_value] = 0
            edges[batch_size * i:batch_size * i + len(distances)] = \
                edges[batch_size * i:batch_size * i + len(distances)] + \
                sparse.lil_matrix(distances_lower)
            del distances_lower
        del distances
    edges = edges + edges.transpose()
    edges_val = edges_val + edges_val.transpose()
    for i, (edge, edge_val) in enumerate(zip(edges, edges_val)):
        edges_val[i, edge.indices] = edge_val.data / edge.data
    edges = edges_val
    edges.setdiag(0)  # remove any left over self edges from patients without any diagnoses (these will be generally matched with others having no diagnoses)
    edges.eliminate_zeros()
    # do upper triangle again and then save
    edges = sparse.tril(edges, k=-1)
    v, u, vals = sparse.find(edges)

    return u, v, vals, k

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--mode', type=str, default='k_closest', help='k_closest or threshold')
    args = parser.parse_args()

    print(args)

    with open('paths.json', 'r') as f:
        eICU_path = json.load(f)["eICU_path"]
        graph_dir = json.load(f)["graph_dir"]

    bert = np.load('{}bert_out.npy'.format(graph_dir))
    u, v, vals, k = make_graph_bert(bert, k=args.k)
    np.savetxt('{}bert_u_k={}_{}.txt'.format(graph_dir, args.k, args.mode), u.astype(int), fmt='%i')
    np.savetxt('{}bert_v_k={}_{}.txt'.format(graph_dir, args.k, args.mode), v.astype(int), fmt='%i')
    np.savetxt('{}bert_scores_k={}_{}.txt'.format(graph_dir, args.k, args.mode), vals.astype(int), fmt='%i')