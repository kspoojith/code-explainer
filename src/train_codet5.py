# src/train_codet5.py
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from src.utils_data import load_codetext_dataset
import torch

MODEL_NAME = "Salesforce/codet5-base"
OUTPUT_DIR = "models/codet5-finetuned"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load data + tokenizer
    train_ds, valid_ds, tokenizer = load_codetext_dataset(MODEL_NAME)

    # Load base model
    print("🔹 Loading CodeT5 base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    # Data collator (handles padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training args
    training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    do_eval=True,
    save_steps=500,
    eval_steps=500,
    save_total_limit=1,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100
)



    # Trainer setup
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🔹 Starting training...")
    trainer.train()
    print("✅ Training complete. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"🎉 Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
