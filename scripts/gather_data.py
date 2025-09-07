#!/usr/bin/env python3
# gather_data.py
# Downloads dataset for Slovene training and splits into train/validation sets

from datasets import load_dataset


def main():
    # Using OSCAR Slovenian dataset as example. You can adjust to your dataset.
    dataset = load_dataset("oscar", "unshuffled_deduplicated_sl", split="train")
    # Shuffle and split dataset
    dataset = dataset.shuffle(seed=42)
    train_val = dataset.train_test_split(test_size=0.05)
    train_dataset = train_val["train"]
    valid_dataset = train_val["test"]
    # Save to disk
    train_dataset.save_to_disk("data/train")
    valid_dataset.save_to_disk("data/valid")


if __name__ == "__main__":
    main()
