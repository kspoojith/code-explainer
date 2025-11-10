# src/utils_data.py
from datasets import load_dataset
from transformers import AutoTokenizer

def load_codetext_dataset(tokenizer_name="Salesforce/codet5-base", max_len=256, sample_size=2000):
    """
    Load a small subset of the CodeXGLUE code-to-text dataset (Python only)
    for quick fine-tuning on CPU.
    """
    print("🔹 Loading dataset (CodeXGLUE code-to-text, Python subset)...")
    ds = load_dataset("code_x_glue_ct_code_to_text", "python")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess(batch):
        inputs = [ex for ex in batch["code"]]
        targets = [ex for ex in batch["docstring"]]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_len)
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=max_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("🔹 Tokenizing dataset...")
    train_ds = ds["train"].select(range(sample_size)).map(preprocess, batched=True)
    valid_ds = ds["validation"].select(range(500)).map(preprocess, batched=True)

    print(f"✅ Dataset ready: {len(train_ds)} train samples, {len(valid_ds)} val samples")
    return train_ds, valid_ds, tokenizer
