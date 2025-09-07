#!/usr/bin/env python3
# preprocess.py
# Tokenizes dataset with a given tokenizer for training an LLM.

import os
from datasets import load_from_disk
from transformers import AutoTokenizer


def main(data_dir="data", tokenizer_name="google/mt5-small", max_length=1024):
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")
    ds_train = load_from_disk(train_path)
    ds_valid = load_from_disk(valid_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds_train = ds_train.map(tokenize, batched=True, remove_columns=ds_train.column_names)
    ds_valid = ds_valid.map(tokenize, batched=True, remove_columns=ds_valid.column_names)
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    ds_train.save_to_disk(os.path.join(processed_dir, "train"))
    ds_valid.save_to_disk(os.path.join(processed_dir, "valid"))


if __name__ == "__main__":
    main()
