"""
Generating bert embeddings from diagnosis text per ICU stay.
"""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import transformers as ppb
import json
from src.utils import load_json


def save_bert_tokens(data_dir, graph_dir, max_len=512):
    """
    read cleaned diagnosis text and tokenize.
    """
    # read diagnosis strings
    dfs = []
    for split in ['train', 'val', 'test']:
        data = pd.read_csv(data_dir + split + '/diagnosis_strings_cleaned.txt', sep="\n", header=None)
        data.columns = ["sentence"]
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    
    # Download pre-trained weights from BERT. This takes a while if running for the first time. 
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    tokenized = df['sentence'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # Cut off tokens at 512 and pad. 
    padded = []
    for i in tokenized.values:
        if len(i) < max_len:
            p = np.array(i + [0]*(max_len-len(i)))
        else:
            p = np.array(i[:max_len])
        padded.append(p)
    padded = np.array(padded)
    attention_mask = np.where(padded != 0, 1, 0)

    # Saving padded.dat and attention_mask.dat
    print('saving data..')
    save_path = graph_dir + 'padded.dat'
    write_file = np.memmap(save_path, dtype=int, mode='w+', shape=(89123, max_len))
    write_file[:, :] = padded
    save_path = graph_dir + 'attention_mask.dat'
    write_file = np.memmap(save_path, dtype=int, mode='w+', shape=(89123, max_len))
    write_file[:, :] = attention_mask
    return padded, attention_mask


def read_data(data_dir, graph_dir, gpu=True):
    """
    read tokens and attention masks as input to BERT model.
    """
    if os.path.exists(graph_dir + 'padded.dat'):
        padded = np.memmap(graph_dir + 'padded.dat', dtype=int, shape=(89123, 512))
        attention_mask = np.memmap(graph_dir + 'attention_mask.dat', dtype=int, shape=(89123, 512))
    else:
        padded, attention_mask = save_bert_tokens(data_dir, graph_dir)
    input_ids = torch.tensor(padded)
    attn_mask = torch.tensor(attention_mask)

    if gpu:
        input_ids = input_ids.to('cuda')
        attn_mask = attn_mask.to('cuda')
    return input_ids, attn_mask


def run_bert_in_mini_batches(graph_dir, model, input_ids, attn_mask, bsz, gpu=True):
    """
    pass the prepared tokens in mini-batches to a pre-trained BERT model and save its output.
    """
    save_path = graph_dir + 'bert_out.npy'
    write_file = np.zeros((89123, 768))
    model.eval()
    if gpu:
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


def main(data_dir, graph_dir, gpu=True):
    """
    generate BERT embeddings from cleaned diagnosis text.
    
    :data_dir: where the diagnosis string files are located
    :graph_dir: where to output the bert embeddings
    """
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    model = model_class.from_pretrained(pretrained_weights)
    input_ids, attn_mask = read_data(data_dir, graph_dir)
    run_bert_in_mini_batches(graph_dir, model, input_ids, attn_mask, bsz=20)


if __name__ == '__main__':
    paths = load_json('paths.json')
    data_dir = paths['eICU_path']
    graph_dir = paths['graph_dir']
    main(data_dir, graph_dir)