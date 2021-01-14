"""
Generating bert embeddings from diagnosis text per ICU stay.
"""
from tqdm import tqdm
import numpy as np
import torch
import transformers as ppb
import json


def read_data(graph_dir):
    padded = np.memmap(graph_dir + 'padded.dat', dtype=int, shape=(89123, 512))
    attention_mask = np.memmap(graph_dir + 'attention_mask.dat', dtype=int, shape=(89123, 512))
    input_ids = torch.tensor(padded).to('cuda')
    attn_mask = torch.tensor(attention_mask).to('cuda')
    return input_ids, attn_mask

def run_bert_in_mini_batches(graph_dir, model, input_ids, attn_mask, bsz):
    save_path = graph_dir + 'bert_out.npy'
    write_file = np.zeros((89123, 768))
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        for i in tqdm(range(0, 89123, bsz)):
            batch_input = input_ids[i: i+bsz]
            batch_mask = attn_mask[i: i+bsz]
            actual_bsz = len(batch_input)
            last_hidden_states = model(batch_input, attention_mask=batch_mask)
            out = last_hidden_states[0][:,0,:].detach().cpu().numpy()
            write_file[i: i+actual_bsz] = out
            if i % 1000 == 0:
                print(out)
    with open(save_path, 'wb') as f:
        np.save(f, write_file)


def main(graph_dir):
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    model = model_class.from_pretrained(pretrained_weights)
    input_ids, attn_mask = read_data(graph_dir)
    run_bert_in_mini_batches(graph_dir, model, input_ids, attn_mask, bsz=20)


if __name__ == '__main__':

    with open('paths.json', 'r') as f:
        graph_dir = json.load(f)["graph_dir"]
    main(graph_dir)