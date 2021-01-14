import random
import json

with open('paths.json', 'r') as f:
    eICU_path = json.load(f)["eICU_path"]
    graph_dir = json.load(f)["graph_dir"]

u = open("{}bert_u_k=5.txt".format(graph_dir), "r")
v = open("{}bert_v_k=5.txt".format(graph_dir), "r")
u_list = u.read()
v_list = v.read()
u_list = u_list.split("\n")
v_list = v_list.split("\n")

diags = []
for file in ['train', 'val', 'test']:
    f = open("{}{}/diagnosis_strings_cleaned.txt".format(eICU_path, file), "r")
    diags.append(f.read().split("\n"))
diags = diags[0][:-1] + diags[1][:-1] + diags[2][:-1]  # the pop is to get rid of the last line of the file

num_to_inspect = 10
edges = list(zip(u_list, v_list))
random.shuffle(edges)

for edge in edges:
    print('patient 1 diags:')
    print(diags[int(edge[0])])
    print('patient 2 diags:')
    print(diags[int(edge[1])])
    num_to_inspect -= 1
    if num_to_inspect < 0:
        break