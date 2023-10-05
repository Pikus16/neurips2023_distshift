import torch
from datasets import load_dataset
from tqdm import tqdm

class SummaryDataset(torch.utils.data.Dataset):
    """Torch wrapper dataset for `openai/summarize_from_feedback` dataset."""

    def __init__(self, split, item_to_ret=None):
        self.hf_dataset = load_dataset('openai/summarize_from_feedback', 'comparisons', split=split)
        self.prompts, self.chosen, self.reject = [], [], []
        for d in tqdm(self.hf_dataset):
            try:
                prompt, chosen, reject = self.process_response(d)
                self.prompts.append(prompt)
                self.chosen.append(chosen)
                self.reject.append(reject)
            except:
                continue
        self.set_dataset_type(item_to_ret)

    def set_dataset_type(self, item_to_ret):
        if item_to_ret == 'prompt':
            self.dataset = self.prompts
        elif item_to_ret == 'chosen':
            self.dataset = self.chosen
        elif item_to_ret == 'rejected':
            self.dataset = self.reject
        else:
            self.dataset = list(zip(self.prompts, self.chosen, self.reject))


    def process_response(self, x):
        prompt = x['info']['post'].strip()
        first = x['summaries'][0]['text'].strip()
        second = x['summaries'][1]['text'].strip()
        choice = x['choice']
        if choice == 0:
            return prompt, first, second
        elif choice == 1:
            return prompt, second, first
        else:
            assert False
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
    
class AnthropicDataset(torch.utils.data.Dataset):
    def __init__(self, split, item_to_ret=None):
        self.hf_dataset = load_dataset('Anthropic/hh-rlhf', split=split)
        self.prompts, self.chosen, self.reject = [], [], []
        for d in tqdm(self.hf_dataset):
            try:
                prompt, chosen, reject = self.process_response(d)
                self.prompts.append(prompt)
                self.chosen.append(chosen)
                self.reject.append(reject)
            except:
                continue
        self.set_dataset_type(item_to_ret)

    def set_dataset_type(self, item_to_ret):
        if item_to_ret == 'prompt':
            self.dataset = self.prompts
        elif item_to_ret == 'chosen':
            self.dataset = self.chosen
        elif item_to_ret == 'rejected':
            self.dataset = self.reject
        else:
            self.dataset = list(zip(self.prompts, self.chosen, self.reject))

    def process_response(self, x):
        chosen = x['chosen']
        reject = x['rejected']
        ind = chosen.rfind('\n\nAssistant:')
        prompt = chosen[:ind].strip()
        assert reject[:len(prompt)] == prompt
        chosen = chosen[ind + len('\n\nAssistant:'):].strip()
        reject = reject[ind + len('\n\nAssistant:'):].strip()
        return prompt, chosen, reject


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]