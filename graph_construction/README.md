Graph Construction
==================================

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