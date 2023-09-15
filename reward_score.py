import pandas as pd
import torch
from utils import get_reward_model, get_scores_df
import fire
import os
def main(path_to_dataframe,
         out_filepath,
         path_to_dataframe_english = None,
         reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2",
         model_max_length=1700,
         device=0):
    df = pd.read_csv(path_to_dataframe).fillna('')
    device =  torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    rank_model, tokenizer = get_reward_model(reward_name=reward_model_name, model_max_length=model_max_length, device=device)
    if path_to_dataframe_english is not None:
        english = pd.read_csv(path_to_dataframe_english).fillna('')
        lang = os.path.splitext(os.path.basename(path_to_dataframe))[0].split('_')[-1] # assumes name is 'english_to_LANG.csv'

        df_ = english.copy()
        df_['prompt'] = df['prompt']
        score_df = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model, device=device)
        score_df.to_csv(os.path.join(out_filepath, f'{lang}_prompt_english_response.csv'), index=False)

        df_ = english.copy()
        df_['chosen'] = df['chosen']
        df_['rejected'] = df['rejected']
        score_df = get_scores_df(df_, tokenizer=tokenizer, rank_model=rank_model, device=device)
        score_df.to_csv(os.path.join(out_filepath, f'english_prompt_{lang}_response.csv'), index=False)
    else:
        score_df = get_scores_df(df, tokenizer=tokenizer, rank_model=rank_model, device=device)
        score_df.to_csv(out_filepath, index=False)

if __name__ == '__main__':
    fire.Fire(main)