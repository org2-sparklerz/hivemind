#!/usr/bin/env python3

""" This script builds a pre-tokenized compressed representation of arxiv-abstract using huggingface/datasets """

import random
from functools import partial
import nltk
from datasets import load_dataset
from transformers import LlamaTokenizerFast

# Define column names for the dataset
COLUMN_NAMES = ("attention_mask", "input_ids", "special_tokens_mask")

def create_instances_from_document(tokenizer, document, max_seq_length):
    """
    Creates training instances from a single document.
    """
    instances = []
    current_chunk = []
    current_length = 0

    # Tokenize document into sentences
    segmented_sents = list(nltk.sent_tokenize(document))

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))

        # Check if current chunk exceeds max sequence length or is last sentence
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 0:
                tokens_a = current_chunk

                # Tokenize and prepare instance
                instance = tokenizer(
                    " ".join(tokens_a),
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )

                instances.append(instance)

            # Reset for next chunk
            current_chunk = []
            current_length = 0

    return instances

def tokenize_function(tokenizer, examples):
    # Remove empty texts
    texts = (text for text in examples["text"] if len(text) > 0 and not text.isspace())
    
    new_examples = {col: [] for col in COLUMN_NAMES}

    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_seq_length=512)
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)

    return new_examples

if __name__ == "__main__":
    random.seed(0)
    nltk.download("punkt")

    # Load LLaMA tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Load arxiv-abstract dataset
    arxiv_abstract = load_dataset("ash001/arxiv-abstract", cache_dir="./data/cache")

    # Tokenize dataset and save to disk
    tokenized_datasets = arxiv_abstract.map(
        partial(tokenize_function, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )

    tokenized_datasets.save_to_disk("./data/llama_tokenized_arxiv")
    tokenizer.save_pretrained("./data/tokenizer")