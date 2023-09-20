import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import random
import os
import pickle
import argparse
from utils import get_reward_model, get_score, get_scores_df, get_percent_chosen
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from urllib.request import urlopen
import logging

np.random.seed(0)
random.seed(0)

word_site = "https://www-personal.umich.edu/~jlawler/wordlist"

response = urlopen(word_site)
txt = response.read().decode()
WORDS = txt.splitlines()


def make_dir(path: str):
    try:
        os.mkdir(path)
    except:
        pass

def get_random_word():
    return random.choice(WORDS)

def insert_word(word_list, i):
    return word_list[:i] + [get_random_word()] + word_list[i:]

def delete_word(word_list, i):
    return word_list[:i] + word_list[i+1:]

def replace_word(word_list, i):
    return word_list[:i] + [get_random_word()] + word_list[i+1:]

def do_action_word(s, i):
    r = np.random.rand()
    if r <= 0.33:
        return ' '.join(w for w in insert_word(s.split(), i)), i + 1
    elif r <= 0.67:
        return ' '.join(w for w in delete_word(s.split(), i)), i
    else:
        return ' '.join(w for w in replace_word(s.split(), i)), i
    
def perturb_word_string(s, prob_perturb=0.1):
    for i in range(len(s.split())):
        r = np.random.rand()
        if r <= prob_perturb:
            s, i = do_action_word(s, i)
    return s


def main(args):
    make_dir(args.path)
    
    original = pd.read_csv('data/openai_summarize_from_feedback/english_original.csv').sample(1000).reset_index(drop=True)
    original = original.fillna('')
    
    logging.info('Perturbing Data')
    perturbed = {}
    for prob_perturb in tqdm(np.arange(0, 1.0, 0.1)):
        perturbed[prob_perturb] = perturb_df(original.copy(), prob_perturb=prob_perturb)
    
    logging.info('Retrieving Reward Model')
    rank_model, tokenizer = get_reward_model()
    
    logging.info('Getting scores from prompt_and_response perturbations')
    perturbed_scores = {}
    scores = {}
    for prob_perturb, df_ in perturbed.items():
        scores[prob_perturb] = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model)
    perturbed_scores['prompt_and_response'] = scores
    
    logging.info('Getting scores from prompt perturbations')
    scores = {}
    for prob_perturb, df_ in perturbed.items():
        df_ = df_.copy()
        df_['prompt'] = original['prompt']
        scores[prob_perturb] = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model)
    perturbed_scores['response'] = scores
    
    logging.info('Getting scores from response perturbations)
    scores = {}
    for prob_perturb, df_ in perturbed.items():
        df_ = df_.copy()
        df_['chosen'] = original['chosen']
        df_['rejected'] = original['rejected']
        scores[prob_perturb] = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model)
    perturbed_scores['prompt'] = scores
                
    
    
    logging.info('Saving scores')
    for perturb_partition in perturbed_scores:
        make_dir(args.path + '/' + perturb_partition)
        for perturb_percent in perturbed_scores[perturb_partition]:
            percent = str(perturb_percent).replace('.','')
            make_dir(f'{args.path}/{perturb_partition}/{percent}')
            pd.DataFrame(perturbed_scores[perturb_partition][perturb_percent]).to_csv(f'{args.path}/{perturb_partition}/{percent}_scores.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='name of the folder path to put results. e.g. trial 2')
    args = parser.parse_args()
    main(args)

