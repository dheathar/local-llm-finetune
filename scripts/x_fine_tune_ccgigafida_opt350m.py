from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def main():
    # Load the ccGigafida dataset (public subset of Gigafida/Gigafida2)
    dataset = load_dataset("cjvt/cc_gigafida", split="train")

    # Choose a compact base model that fits on GPUs like the RTX 4060
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenization function to flatten paragraphs into a single string
    def tokenize(batch):
        texts = ["\n\n".join(paragraphs) for paragraphs in batch["doc_string"]]
        return tokenizer(texts, truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments tuned for a small GPU
    training_args = TrainingArguments(
        output_dir="outputs/opt350m-ccgigafida",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        evaluation_strategy="steps",
        save_strategy="epoch",
        logging_steps=100,
        fp16=True,
    )

    # Create Trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized.shuffle(seed=42).select(range(2000)),
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("outputs/opt350m-ccgigafida")


if __name__ == "__main__":
    main()
