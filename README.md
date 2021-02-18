Predicting Patient Outcomes with Graph Representation Learning
===============================

This repository contains the code used for Predicting Patient Outcomes with Graph Representation Learning. You can watch a video of the spotlight talk at W3PHIAI (AAAI workshop) here:

[![Watch the video](https://img.youtube.com/vi/Q_VrAYL8Tho/maxresdefault.jpg)](https://www.youtube.com/watch?v=Q_VrAYL8Tho)
 
## Citation
If you use this code or the models in your research, please cite the following:

```
@misc{rocheteautong2021,
      title={Predicting Patient Outcomes with Graph Representation Learning}, 
      author={Emma Rocheteau and Catherine Tong and Petar Veličković and Nicholas Lane and Pietro Liò},
      year={2021},
      eprint={2101.03940},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Motivation
Recent work on predicting patient outcomes in the Intensive Care Unit (ICU) has focused heavily on the physiological time series data, largely ignoring sparse data such as diagnoses and medications. When they are included, they are usually concatenated in the late stages of a model, which may struggle to learn from rarer disease patterns. Instead, we propose a strategy to exploit diagnoses as relational information by connecting similar patients in a graph. To this end, we propose LSTM-GNN for patient outcome prediction tasks: a hybrid model combining Long Short-Term Memory networks (LSTMs) for extracting temporal features and Graph Neural Networks (GNNs) for extracting the patient neighbourhood information. We demonstrate that LSTM-GNNs outperform the LSTM-only baseline on length of stay prediction tasks on the eICU database. More generally, our results indicate that exploiting information from neighbouring patient cases using graph neural networks is a promising research direction, yielding tangible returns in supervised learning performance on Electronic Health Records.


## Pre-Processing Instructions

### eICU Pre-Processing

1) To run the sql files you must have the eICU database set up: https://physionet.org/content/eicu-crd/2.0/. 

2) Follow the instructions: https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ to ensure the correct connection configuration. 

3) Replace the eICU_path in `paths.json` to a convenient location in your computer, and do the same for `eICU_preprocessing/create_all_tables.sql` using find and replace for 
`'/Users/emmarocheteau/PycharmProjects/eICU-GNN-LSTM/eICU_data/'`. Leave the extra '/' at the end.

4) In your terminal, navigate to the project directory, then type the following commands:

    ```
    psql 'dbname=eicu user=eicu options=--search_path=eicu'
    ```
    
    Inside the psql console:
    
    ```
    \i eICU_preprocessing/create_all_tables.sql
    ```
    
    This step might take a couple of hours.
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 -m eICU_preprocessing.run_all_preprocessing
    ```
    
### Graph Construction

To make the graphs, you can use the following scripts:

This is to make most of the graphs that we use. You can alter the arguments given to this script.
```
python3 -m graph_construction.create_graph --freq_adjust --penalise_non_shared --k 3 --mode k_closest
```
Write the diagnosis strings into `eICU_data` folder:
```
python3 -m graph_construction.get_diagnosis_strings
```
Get the bert embeddings:
```
python3 -m graph_construction.bert
```
Create the graph from the bert embeddings:
```
python3 -m graph_construction.create_bert_graph --k 3 --mode k_closest
```


Alternatively, you can request to download our graphs using this link:
https://drive.google.com/drive/folders/1yWNLhGOTPhu6mxJRjKCgKRJCJjuToBS4?usp=sharing

## Training the ML Models

Before proceeding to training the ML models, do the following.

1) Define data_dir, graph_dir, log_path and ray_dir in `paths.json` to convenient locations.

2) Run the following to unpack the processed eICU data into mmap files for easy loading during training. The mmap files will be saved in `data_dir`.
    ```
    python3 -m src.dataloader.convert
    ```

The following commands train and evaluate the models introduced in our paper.

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
