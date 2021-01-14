import pandas as pd
import json
import copy

with open('paths.json', 'r') as f:
    eICU_path = json.load(f)["eICU_path"]
    graph_dir = json.load(f)["graph_dir"]

u = open("{}bert_u_k=5.txt".format(graph_dir), "r")
v = open("{}bert_v_k=5.txt".format(graph_dir), "r")
u_list = u.read()
v_list = v.read()
u_list = u_list.split("\n")
v_list = v_list.split("\n")

train_labels = pd.read_csv('{}train/labels.csv'.format(eICU_path), index_col='patient')
val_labels = pd.read_csv('{}val/labels.csv'.format(eICU_path), index_col='patient')
test_labels = pd.read_csv('{}test/labels.csv'.format(eICU_path), index_col='patient')
all_labels = pd.concat([train_labels, val_labels, test_labels], sort=False)

both_died_outcome = 0
both_survived_outcome = 0
diff_outcome = 0
num_comparisons = 100000
tracker = copy.copy(num_comparisons)
for edge in zip(u_list, v_list):
    if all_labels.iloc[int(edge[0])]['actualhospitalmortality'] == all_labels.iloc[int(edge[1])]['actualhospitalmortality']:
        if all_labels.iloc[int(edge[0])]['actualhospitalmortality'] == 1:
            both_died_outcome += 1
        else:
            both_survived_outcome += 1
    else:
        diff_outcome += 1
    tracker -= 1
    if tracker < 0:
        break

perc_both_died = both_died_outcome/num_comparisons * 100
perc_both_survived = both_survived_outcome/num_comparisons * 100
perc_died_and_survived = diff_outcome/num_comparisons * 100
print('==> GRAPH IN QUESTION')
print(str(perc_both_died)[:4] + '% of the connections both died')
print(str(perc_both_survived)[:4] + '% of the connections both survived')
print(str(perc_died_and_survived)[:4] + '% of the connections had one death and one survival')
perc_involving_died = perc_both_died + perc_died_and_survived * 0.5
print(str(perc_involving_died)[:4] + '% of the nodes involved in edges have died')

probs = all_labels['actualhospitalmortality'].value_counts()/len(all_labels)
perc_both_died = probs[1]**2 * 100
perc_both_survived = probs[0]**2 * 100
perc_died_and_survived = probs[0] * probs[1] * 100
print('==> RANDOM GRAPH')
print(str(perc_both_died)[:4] + '% of the connections both died')
print(str(perc_both_survived)[:4] + '% of the connections both survived')
print(str(perc_died_and_survived)[:4] + '% of the connections had one death and one survival')
perc_involving_died = perc_both_died + perc_died_and_survived * 0.5
print(str(perc_involving_died)[:4] + '% of the nodes involved in edges have died')