# neurips2023_distshift

## Data
### OpenAI Summarize From Feedback
We took the data from [OpenAI's Summarize From Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) data and translated prompts and each response. We then ran (OpenAssistant's Deberta Reward model)(https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) on prompts and responses.
* [Original dataset in English](https://drive.google.com/file/d/1HuL0bVM5P7DnLuOm5VaAKEjtUaT_HAjF/view?usp=drive_link)
* [English translated to Chinese](https://drive.google.com/file/d/1E-14h_ZKxTmLlSbwAhOK9pK6dNeJOBF6/view?usp=drive_link) with [this model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
* [English translated to Chinese back to English](https://drive.google.com/file/d/12UIGJXfxMeVseYIpH7oCeK8LZ8q4ROy-/view?usp=sharing) with [this model](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
   

## Reward Scores
On the [OpenAI's Summarize From Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) dataset, with translated prompts / responses as listed above, we used the [OpenAssistant's Deberta Reward model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) to get scores for each prompt/response pair. These can all be found in [model_scores/deberta_v3_large](model_scores/deberta_v3_large). We have the [original scores](model_scores/deberta_v3_large/english_original_scores.csv). Additionally, for each language, we have 4 scores: both the prompt and response in the translated language, just the prompt translated, just the response translated, and then both the prompt and response translated back to english.