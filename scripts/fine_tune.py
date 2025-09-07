#!/usr/bin/env python3
# fine_tune.py
# Fine-tunes a causal language model on processed Slovenian dataset.

import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling


def main(data_dir="data/processed", model_name="bigscience/bloom-560m", output_dir="outputs/finetuned_model", epochs=1, batch_size=1):
    train_ds = load_from_disk(os.path.join(data_dir, "train"))
    eval_ds = load_from_disk(os.path.join(data_dir, "valid"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
