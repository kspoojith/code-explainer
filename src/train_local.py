# src/train_local.py
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from src.utils_local_dataset import load_local_explain_dataset

BASE_MODEL = "Salesforce/codet5-base"         # always load base model
OUTPUT_DIR = "models/codet5-finetuned"
overwrite_output_dir = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load local dataset
    train_ds, val_ds, tokenizer = load_local_explain_dataset(
        path="data/raw/code_explainer_dataset.jsonl",
        tokenizer_name=BASE_MODEL,
    )

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=20,
        save_total_limit=1,
        fp16=False,
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🔧 Training...")
    trainer.train()

    print("💾 Saving...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"🎉 Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
