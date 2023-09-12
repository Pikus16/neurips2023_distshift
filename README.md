# neurips2023_distshift

## Data
### OpenAI Summarize From Feedback
We took the data from [OpenAI's Summarize From Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) data and translated prompts and each response. We then ran (OpenAssistant's Deberta Reward model)(https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) on prompts and responses.
* [Original dataset in English](https://drive.google.com/file/d/1HuL0bVM5P7DnLuOm5VaAKEjtUaT_HAjF/view?usp=drive_link)
* [English translated to Chinese](https://drive.google.com/file/d/1E-14h_ZKxTmLlSbwAhOK9pK6dNeJOBF6/view?usp=drive_link) with [this model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
* [English translated to Chinese back to English](https://drive.google.com/file/d/12UIGJXfxMeVseYIpH7oCeK8LZ8q4ROy-/view?usp=sharing) with [this model](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
   

## Reward Scores
On the [OpenAI's Summarize From Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) dataset, with translated prompts / responses as listed above, we used the [OpenAssistant's Deberta Reward model](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) to get scores for each prompt/response pair.
* [Original scores in English](https://drive.google.com/file/d/1WyFdTS9gTKoIQ8UyKqcEaXS824ZUKyCu/view?usp=drive_link)
* [English translated to Chinese scores](https://drive.google.com/file/d/19iO6UNgVrCg_gEo-3YDH5dxPANqQQAsO/view?usp=drive_link)
* [English translated to Chinese back to Englis Scores](https://drive.google.com/file/d/1Jybkwlh7HVg7MRJan8LFnhm2E-INBvGS/view?usp=drive_link)
* [Original English Prompts with Chinese Responses](https://drive.google.com/file/d/1DV2KOQ_SZMeyAkFGpP3PuNI_Lb04YG3u/view?usp=drive_link)
* [Chinese Prompts with Original English Responses](https://drive.google.com/file/d/1LrT9ToX7s7J6rTuhN8Ih9p5rcc_C9VhZ/view?usp=drive_link)