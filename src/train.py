# src/train.py
import os
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from preprocess import prepare_example

MODEL_NAME = "Salesforce/codet5-base"  # baseline code summarization model
TOKENIZER_NAME = MODEL_NAME
OUTPUT_DIR = "models/codet5-finetuned"

def preprocess_dataset(ds, tokenizer, max_input_len=512, max_target_len=128):
    def _map(ex):
        prep = prepare_example(ex)
        inputs = tokenizer(prep['input'], truncation=True, max_length=max_input_len)
        targets = tokenizer(prep['target'], truncation=True, max_length=max_target_len)
        inputs['labels'] = targets['input_ids']
        return inputs
    return ds.map(_map, remove_columns=ds.column_names)

def main():
    # load dataset - using CodeXGLUE python split here
    ds = load_dataset("google/code_x_glue_ct_code_to_text", "python")
    # pick small subset for quick testing (remove these lines for full training)
    train_ds = ds['train'].shuffle(seed=42).select(range(2000))
    val_ds = ds['validation'].shuffle(seed=42).select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset = preprocess_dataset(train_ds, tokenizer)
    val_dataset = preprocess_dataset(val_ds, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_total_limit=2,
        fp16=False  # set to True if running on GPU with mixed precision
    )

    # Simple metric using rouge
    rouge = load_metric("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        # return a simplified dict
        return {k: float(v.mid.fmeasure * 100) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
