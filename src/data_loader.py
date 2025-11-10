# src/data_loader.py
from datasets import load_dataset, DatasetDict
import os
import json

def load_codexglue(split='train', local_path=None):
    """
    Load CodeXGLUE-like dataset. If local_path is provided, load JSONL from local.
    Otherwise expects you to have HF dataset downloaded locally / via 'datasets'.
    """
    if local_path:
        # Expect local_path to be a jsonl file with {"code": "...", "docstring": "..."} per line
        rows = []
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        return DatasetDict({'train': rows})  # simple fallback
    else:
        # Example: using code_x_glue_ct_code_to_text - adjust as needed
        ds = load_dataset("google/code_x_glue_ct_code_to_text", "python")
        return ds

def preview_sample(dataset, split='train', n=3):
    d = dataset[split]
    for i in range(n):
        print("=== SAMPLE ===")
        print("CODE:\n", d[i]['code'][:400])
        print("DOCSTRING:\n", d[i]['docstring'])
