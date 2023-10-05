"""Converts a hugging face dataset to csv"""
import pandas as pd
import fire
from dataset_util import SummaryDataset, AnthropicDataset

def main(output_filepath, dataset_name, split='test'):
    if dataset_name == 'openai_summarize_from_feedback':
        dataset = SummaryDataset(split=split, item_to_ret=None)
    elif dataset_name == 'anthropic_hh_rlhf':
        dataset = AnthropicDataset(split=split, item_to_ret=None)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    
    assert len(dataset[0]) == 3
    for i in range(3):
        assert isinstance(dataset[0][i], str)

    entries = []
    for d in dataset:
        entries.append({
            'prompt' : d[0],
            'chosen' : d[1],
            'rejected' : d[2]
        })
    df = pd.DataFrame(entries)
    print(f'Writing dataset of length {len(df)} to {output_filepath}')
    df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    fire.Fire(main)