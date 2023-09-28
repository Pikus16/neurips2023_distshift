from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

class RewardModelWrapper:

    def __init__(self, reward_name,
                 model_max_length=1700,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.reward_name = reward_name
        self.model_max_length = model_max_length
        self.device = device
        self.rank_model, self.tokenizer = self.get_reward_model(reward_name=self.reward_name,
                                                           model_max_length=self.model_max_length,
                                                           device=self.device)
        
    def format_shp_prompt(self, prompt, response):
        if random.random() <= 0.5:
            response_a = response
            response_b = "."
            expected = 71 #'A'
        else:
            response_a = "."
            response_b = response
            expected = 272 #'B'
        s = f"POST: {prompt} \n\n RESPONSE A: {response_a} \n\n RESPONSE B: {response_b} \n\n Which response is better? RESPONSE"
        return s, expected
    
    def get_reward_model(self, reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2",
                        model_max_length=1700,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if reward_name == "OpenAssistant/reward-model-deberta-v3-large-v2":
            rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(reward_name, model_max_length=model_max_length)
        elif reward_name == 'stanfordnlp/SteamSHP-flan-t5-xl':
            rank_model = T5ForConditionalGeneration.from_pretrained(reward_name).to(device)
            tokenizer = T5Tokenizer.from_pretrained(reward_name)
        else:
            raise ValueError(f'Unknown reward model {reward_name}')
        return rank_model, tokenizer

    def get_score(self, prompt, response):
        if self.reward_name == "OpenAssistant/reward-model-deberta-v3-large-v2":
            inputs = self.tokenizer(prompt, response, return_tensors='pt', truncation=True)
            inputs = inputs.to(self.device)
            return self.rank_model(**inputs).logits[0].cpu().detach().item()
        elif self.reward_name == 'stanfordnlp/SteamSHP-flan-t5-xl':
            input_text, expected = self.format_shp_prompt(prompt, response)
            x = self.tokenizer([input_text], return_tensors='pt').input_ids.to(self.device)
            outputs = self.ramk_model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
            return outputs.scores[0][:, expected].item()
        else:
            raise ValueError(f'Unknown reward model {self.reward_name}')

    def get_scores_df(self, df):
        chosen_scores = []
        reject_scores = []
        existing = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            if (row.prompt, row.chosen) not in existing:
                existing[(row.prompt, row.chosen)] = self.get_score(row.prompt, row.chosen)
            chosen_scores.append(existing[(row.prompt, row.chosen)])

            if (row.prompt, row.rejected) not in existing:
                existing[(row.prompt, row.rejected)] = self.get_score(row.prompt, row.rejected)
            reject_scores.append(existing[(row.prompt, row.rejected)])
        return pd.DataFrame({'chosen' : chosen_scores, 'rejected' : reject_scores})

def get_percent_chosen(df, chosen_col = 0):
    return np.mean(df.values.argmax(axis=1) == chosen_col)
