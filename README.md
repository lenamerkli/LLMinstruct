---
configs:
  - config_name: default
    data_files:
      - split: train
        path: data.parquet
license:
  - apache-2.0
language:
  - multilingual
  - de
  - en
tags:
  - chat
  - instruction-tuning
task_categories:
  - text-generation
---

# Dataset Card for lenamerkli/LLMinstruct

This dataset consists of instruct finetuning data from all of my projects.

## Dataset Details

### Dataset Description

- **Curated by:** Lena Merkli
- **Languages (NLP):** Mostly english and german
- **License:** Apache license 2.0

### Dataset Sources

- **Repository:** https://github.com/lenamerkli/LLMinstruct

## Uses

This dataset is useful for instruct-tuning or fine-tuning large language models.

### Use Recommendations

I recommend to use only the following data for training:

- all data marked as containing no mistakes
- the drawback chess data
- all moral data

## Dataset Structure

### Data Fields

- `messages`: List of conversation messages, each containing:
  - `role`: Either "user", "assistant", "tool" or "system"
  - `content`: The text content of the message
  - `attachments`: List of attachments (e.g. images referenced by SHA256 hash, tool definitions)
- `project`: String identifier for the source project
- `synthetic`: Boolean indicating whether the data was synthetically generated
- `mistakes`: Boolean indicating whether the data may contain errors
- `languages`: List of detected languages; not accurate
- `token_count`: Integer count of tokens in the conversation excluding attachments using the Apertus tokenizer

### Data Splits

The dataset contains a single `train` split stored in `data.parquet`.

### Source Projects

| Project            | Synthetic | Mistakes | Description                      |
|--------------------|-----------|----------|----------------------------------|
| biasbench          | ✓         | (✓)*     | Instruction-following evaluation |
| infinite_craft     |           |          | Game-related instructions        |
| misc               | (✓)**     | (✓)*     | Miscellaneous instructions       |
| moral              | ✓         | (✓)*     | Moral/ethical reasoning          |
| drawback_chess     | ✓         | ✓        | Chess with tool calling          |
| explain_meme       | ✓         | ✓        | Meme explanation with images     |
| ingredient_scanner | (✓)**     | (✓)*     | Ingredient analysis              |
| topic_categorizer  | ✓         |          | Topic classification             |

*only parts contain mistakes

**only partially synthetic

## Personal and Sensitive Information

The dataset should be free of personal and sensitive information. Included in the dataset is publicly available information about public figures as well as fake names.

## Bias, Risks, and Limitations

Some data contains mistakes. This is labeled as such in the `mistakes` column. The dataset contains biases of the authors.
