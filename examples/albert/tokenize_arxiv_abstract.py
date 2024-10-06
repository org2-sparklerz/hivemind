#!/usr/bin/env python3

""" This script builds a pre-tokenized compressed representation of arxiv-abstract dataset using huggingface/datasets """

import random
from functools import partial
import nltk
from datasets import load_dataset
from transformers import LlamaTokenizer

COLUMN_NAMES = ("attention_mask", "input_ids", "labels")

def create_instances_from_document(tokenizer, document, max_seq_length):
    """
    Creates training instances from a single document.
    Reuses code from the original ALBERT implementation (Google AI, 2018)
    https://github.com/google-research/albert/blob/master/create_pretraining_data.py#L267
    """
    instances = []
    current_chunk = []
    current_length = 0
    segmented_sents = list(nltk.sent_tokenize(document))

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 1:
                instance = tokenizer(
                    " ".join(current_chunk),
                    truncation=True,
                    max_length=max_seq_length,
                    padding="max_length",
                )
                
                assert len(instance["input_ids"]) == max_seq_length
                instance["labels"] = instance["input_ids"].copy()
                instances.append(instance)
                
                current_chunk = []
                current_length = 0
                
    return instances

def tokenize_function(tokenizer, examples):
    # Remove empty texts
    texts = (text for text in examples["abstract"] if len(text) > 0 and not text.isspace())
    
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
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    arxiv_abstract = load_dataset("ash001/arxiv-abstract")
    
    tokenized_datasets = arxiv_abstract.map(
        partial(tokenize_function, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=["abstract"],
    )
    
    tokenized_datasets.save_to_disk("./data/llama_tokenized_arxiv_abstract")
    tokenizer.save_pretrained("./data/tokenizer")