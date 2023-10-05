import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import random
from utils import get_reward_model, get_score, get_scores_df, get_percent_chosen
import matplotlib.pyplot as plt
import pickle

def main():
    np.random.seed(0)
    random.seed(0)

    original = pd.read_csv('data/openai_summarize_from_feedback/english_original.csv').reset_index(drop=True)
    original = original.fillna('')

    chars = string.ascii_letters + '0123456789 \n'
    def get_random_char():
        return random.choice(chars)

    def insert(s, i):
        return s[:i] + get_random_char() + s[i:]

    def delete(s, i):
        return s[:i] + s[i+1:]

    def replace(s, i):
        return s[:i] + get_random_char() + s[i+1:]

    def do_action(s, i):
        r = np.random.rand()
        if r <= 0.33:
            return insert(s, i), i+ 1
        elif r <= 0.67:
            return delete(s,i), i
        else:
            return replace(s,i), i
        
    def perturb_string(s, prob_perturb=0.1):
        for i in range(len(s)):
            r = np.random.rand()
            if r <= prob_perturb:
                s, i = do_action(s, i)
        return s

    def perturb_df(df, prob_perturb):
        for i, row in original.iterrows():
            df.prompt.iloc[i] = perturb_string(row.prompt, prob_perturb=prob_perturb)
            df.iloc[i].chosen = perturb_string(row.chosen, prob_perturb=prob_perturb)
            df.iloc[i].rejected = perturb_string(row.rejected, prob_perturb=prob_perturb)
        return df

    rank_model, tokenizer = get_reward_model()

    res = []
    for iter in range(10):
        perturbed = {}
        for prob_perturb in tqdm(np.arange(0, 1.0, 0.1)):
            perturbed[prob_perturb] = perturb_df(original.copy(), prob_perturb=prob_perturb)

        perturbed_scores = {}
        scores = {}
        for prob_perturb, df_ in perturbed.items():
            scores[prob_perturb] = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model)
        perturbed_scores['prompt_and_response'] = scores

        scores = {}
        for prob_perturb, df_ in perturbed.items():
            df_ = df_.copy()
            df_['prompt'] = original['prompt']
            scores[prob_perturb] = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model)
        perturbed_scores['response'] = scores

        scores = {}
        for prob_perturb, df_ in perturbed.items():
            df_ = df_.copy()
            df_['chosen'] = original['chosen']
            df_['rejected'] = original['rejected']
            scores[prob_perturb] = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model)
        perturbed_scores['prompt'] = scores

        probs_both, accs_both = [], []
        for k,v in perturbed_scores['prompt_and_response'].items():
            probs_both.append(k)
            accs_both.append(get_percent_chosen(v))
        probs_prompt, accs_prompt = [], []
        for k,v in perturbed_scores['prompt'].items():
            probs_prompt.append(k)
            accs_prompt.append(get_percent_chosen(v))
        probs_response, accs_response = [], []
        for k,v in perturbed_scores['response'].items():
            probs_response.append(k)
            accs_response.append(get_percent_chosen(v))

        res.append(
            {
                'accs_both': accs_both,
                'accs_prompt': accs_prompt,
                'accs_response': accs_response,
                'probs_both': probs_both,
                'probs_prompt': probs_prompt,
                'probs_response': probs_response,
            }
        )
        with open('res_perturb_chars.pkl', 'wb') as f:
            pickle.dump(res, f)
    
if __name__ == '__main__':
    main()