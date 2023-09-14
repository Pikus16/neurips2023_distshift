from datasets import load_dataset
from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import fire

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, l):
        self.og_list = np.array(l)
        self.map_value_to_ind = defaultdict(list)
        for i, x in enumerate(self.og_list):
            self.map_value_to_ind[x].append(i)
        self.vals = np.array(list(self.map_value_to_ind.keys()))

    def __len__(self):
        return len(self.vals)
    
    def __getitem__(self, idx):
        return self.vals[idx]
    
    def to_list(self, list_to_insert):
        res = [None] * len(self.og_list)
        for i, vs in enumerate(self.map_value_to_ind.values()):
            for v in vs:
                res[v] = list_to_insert[i]
        for r in res:
            assert r is not None
        return res
    
def break_up_sentences(sentences):
    sentence_list = []
    sentence_to_prompt = []
    for i,x in enumerate(sentences):
        sents = sent_tokenize(x)
        sentence_list.extend(sents)
        sentence_to_prompt.extend([i] * len(sents))
    assert len(sentence_to_prompt) == len(sentence_list)
    return sentence_list, np.array(sentence_to_prompt)
    
def get_translation_model(model_name, device=0, truncation=True):
    return pipeline("translation", model=model_name, device=device, truncation=truncation)

def do_translation(pipe_translate, dataset, batch_size=64):
    all_results = []
    for out in tqdm(pipe_translate(dataset, batch_size=batch_size), total=len(dataset)):
        all_results.append(out[0]['translation_text'])
    return all_results

def stich_back_sentences(dataset, sentence_to_prompt, translated):
    sentences = np.array(dataset.to_list(translated))
    s = []
    for i in np.arange(max(sentence_to_prompt) + 1):
        inds = np.where(sentence_to_prompt == i)[0]
        s.append(' '.join(sentences[inds]))
    return s

def main(path_to_dataframe,
         model_to_translate,
         out_filepath,
         batch_size = 64,
         device=0):
    df = pd.read_csv(path_to_dataframe).fillna('')
    pipe_translate = get_translation_model(model_to_translate, device=device)
    all_result_types = {}
    for text_type in ['prompt', 'chosen', 'rejected']:
        sentence_list, sentence_to_prompt = break_up_sentences(df[text_type])
        dataset = ListDataset(sentence_list)
        translated = do_translation(pipe_translate, dataset, batch_size=batch_size)
        all_result_types[text_type] = stich_back_sentences(dataset, sentence_to_prompt, translated)
    all_result_types = pd.DataFrame(all_result_types)
    all_result_types.to_csv(out_filepath, index=False)
        


if __name__ == '__main__':
    fire.Fire(main)