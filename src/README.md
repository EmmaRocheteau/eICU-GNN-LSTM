Training the ML models
==================================

Before proceeding to training the ML models, do the following.

1) Define data_dir, graph_dir, log_path, and ray_dir in `paths.json` to convenient locations.

2) Run the following to unpack the processed eICU data into mmap files for easy loading during training. The mmap files will be saved in `data_dir`.
    ```
    python3 -m src.dataloader.convert
    ```

The following commands trains and evaluates the models introduced in our paper.

N.B.

- The models are structured using pytorch-lightning. Graph neural networks and neighbourhood sampling are implemented using pytorch-geometric.

- Our models assume a default graph which is made with k=3 under a k-closest scheme. If you wish to use other graphs, refer to `read_graph_edge_list` in `src/dataloader/pyg_reader.py` to add a reference handle to `version2filename` for your graph. 

- The default task is **In-House-Mortality Prediction (ihm)**, add `--task los` to the command to perform the **Length-of-Stay Prediction (los)** task instead. 

- These commands use the best set of hyperparameters; To use other hyperparameters, remove `--read_best` from the command and refer to `src/args.py`. 

### a. LSTM-GNN
The following runs the training and evaluation for LSTM-GNN models. `--gnn_name` can be set as `gat`, `sage`, or `mpnn`. When `mpnn` is used, add `--ns_sizes 10` to the command.

```
python3 -m train_ns_lstmgnn --bilstm --ts_mask --add_flat --class_weights --gnn_name gat --add_diag --read_best
```

The following runs a hyperparameter search.

```
python3 -m src.hyperparameters.lstmgnn_search --bilstm --ts_mask --add_flat --class_weights  --gnn_name gat --add_diag
```

### b. Dynamic LSTM-GNN
The following runs the training & evaluation for dynamic LSTM-GNN models. `--gnn_name` can be set as `gcn`, `gat`, or `mpnn`.

```
python3 -m train_dynamic --bilstm --random_g --ts_mask --add_flat --class_weights --gnn_name mpnn --read_best
```

The following runs a hyperparameter search.

```
python3 -m src.hyperparameters.dynamic_lstmgnn_search --bilstm --random_g --ts_mask --add_flat --class_weights --gnn_name mpnn
```

### c. GNN
The following runs the GNN models (with neighbourhood sampling). `--gnn_name` can be set as `gat`, `sage`, or `mpnn`. When `mpnn` is used, add `--ns_sizes 10` to the command.

```
python3 -m train_ns_gnn --ts_mask --add_flat --class_weights --gnn_name gat --add_diag --read_best
```

The following runs a hyperparameter search.

```
python3 -m src.hyperparameters.ns_gnn_search --ts_mask --add_flat --class_weights --gnn_name gat --add_diag
```

### d. LSTM (Baselines)
The following runs the baseline bi-LSTMs. To remove diagnoses from the input vector, remove `--add_diag` from the command.
```
python3 -m train_ns_lstm --bilstm --ts_mask --add_flat --class_weights --num_workers 0 --add_diag --read_best
```

The following runs a hyperparameter search.

```
python3 -m src.hyperparameters.lstm_search --bilstm --ts_mask --add_flat --class_weights --num_workers 0 --add_diag
```
