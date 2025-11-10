# src/utils_local_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

def load_local_explain_dataset(
    path="data/raw/code_explainer_dataset.jsonl",
    tokenizer_name="Salesforce/codet5-base",
    max_source_len=384,
    max_target_len=160,
    val_size=0.1,
    seed=42,
):
    # Load JSONL
    ds = load_dataset("json", data_files=path, split="train")

    if "code" not in ds.column_names or "explanation" not in ds.column_names:
        raise ValueError(f"Dataset must contain 'code' and 'explanation' fields.\nFound: {ds.column_names}")

    codes = ds["code"]
    expls = ds["explanation"]

    # Split indices
    train_idx, val_idx = train_test_split(
        list(range(len(codes))), test_size=val_size, random_state=seed
    )

    # Load correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)

    prefix = "explain code: "

    def encode_indices(indices):
        src = [prefix + codes[i] for i in indices]
        tgt = [expls[i] for i in indices]

        model_inputs = tokenizer(
            src, max_length=max_source_len, padding="max_length", truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt, max_length=max_target_len, padding="max_length", truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Torch wrapper
    class Seq2SeqDataset(Dataset):
        def __init__(self, idxs):
            encoded = encode_indices(idxs)
            self.data = {k: torch.tensor(v) for k, v in encoded.items()}

        def __len__(self):
            return len(self.data["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    return Seq2SeqDataset(train_idx), Seq2SeqDataset(val_idx), tokenizer
