# src/preprocess.py
import re

def clean_code(code: str) -> str:
    # basic cleaning: remove excessive whitespace
    code = code.strip()
    code = re.sub(r'\s+\n', '\n', code)
    return code

def prepare_example(example):
    # the model will receive code as input and docstring as target
    code = clean_code(example.get('code', ''))
    doc = example.get('docstring', '') or example.get('doc', '') or ""
    if not doc:
        # fallback: maybe use first comment or path
        doc = example.get('description', '')
    return {'input': code, 'target': doc.strip()}
