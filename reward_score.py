import pandas as pd
import torch
from utils import get_reward_model, get_scores_df
import fire

def main(path_to_dataframe,
         out_filepath,
         reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2",
         model_max_length=1700,
         device=0):
    df = pd.read_csv(path_to_dataframe).fillna('')
    device =  torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    rank_model, tokenizer = get_reward_model(reward_name=reward_model_name, model_max_length=model_max_length, device=device)
    score_df = get_scores_df(df, tokenizer=tokenizer, rank_model=rank_model, device=device)
    score_df.to_csv(out_filepath, index=False)

if __name__ == '__main__':
    fire.Fire(main)