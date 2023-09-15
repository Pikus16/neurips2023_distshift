# neurips2023_distshift

## Data
For each dataset, we translated prompts and each response. We then translated back to English.

### [OpenAI Summarize From Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
We translated the validation set (about 86k examples).The original dataset can be found [here](https://drive.google.com/file/d/1HuL0bVM5P7DnLuOm5VaAKEjtUaT_HAjF/view?usp=drive_link). All translations can be found [here](https://drive.google.com/drive/folders/1y8rW85yvKEpwKEadlumSV23p_FGxiJsO?usp=drive_link).

### [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
We translated the test set for samples where the score ratio is >= 2. The original dataset can be found [here](https://drive.google.com/file/d/1kfA_DoLmp6NBf8Zif1h0982mlHCSf4lG/view?usp=drive_link). All translations can be found [here](https://drive.google.com/drive/folders/1t_I9gbIBBbWy6f2dpEag7oL2uVAhDOfa?usp=drive_link).
   

## Reward Scores
For each of the above datasets, with translated prompts / responses as listed above, we used the [OpenAssistant's Deberta Reward model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) to get scores for each prompt/response pair. These can all be found in [model_scores/deberta_v3_large](model_scores/deberta_v3_large). We have the original scores for each dataset. Additionally, for each language, we have 4 scores: both the prompt and response in the translated language, just the prompt translated, just the response translated, and then both the prompt and response translated back to english.