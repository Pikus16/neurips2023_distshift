from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_reward_model(reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2",
                     model_max_length=1700,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(reward_name, model_max_length=model_max_length)
    return rank_model, tokenizer

def get_score(prompt, response, tokenizer, rank_model,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    inputs = tokenizer(prompt, response, return_tensors='pt', truncation=True)
    inputs = inputs.to(device)
    return rank_model(**inputs).logits[0].cpu().detach().item()

def get_scores_df(df, tokenizer, rank_model,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    chosen_scores = []
    reject_scores = []
    existing = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if (row.prompt, row.chosen) not in existing:
            existing[(row.prompt, row.chosen)] = get_score(row.prompt, row.chosen, tokenizer=tokenizer, rank_model=rank_model, device=device)
        chosen_scores.append(existing[(row.prompt, row.chosen)])

        if (row.prompt, row.rejected) not in existing:
            existing[(row.prompt, row.rejected)] = get_score(row.prompt, row.rejected, tokenizer=tokenizer, rank_model=rank_model, device=device)
        reject_scores.append(existing[(row.prompt, row.rejected)])
    return pd.DataFrame({'chosen' : chosen_scores, 'rejected' : reject_scores})

def get_percent_chosen(df, chosen_col = 0):
    return np.mean(df.values.argmax(axis=1) == chosen_col)
