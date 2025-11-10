# 🇮🇳 Telugu Code Explainer

A lightweight system that explains source code in English and then translates the final explanation into Telugu.  
Built using **FastAPI**, **Streamlit**, a fine-tuned **CodeT5 model**, and a **fast rule-based Telugu translator**.

---

## 🚀 Features
- Code explanation using a fine-tuned CodeT5 model  
- Clean English explanation  
- Telugu explanation using instant rule-based translation  
- FastAPI backend  
- Streamlit frontend  
- Very fast inference with no heavy translation models  
- Minimal and production-friendly design  

---

## 📁 Project Structure

```
telugu-code-explainer/
│
├── src/
│ ├── inference.py
│ ├── preprocess.py
│ ├── translate.py
│ ├── streamlit_app.py
│ ├── train.py
│ ├── train_local.py
│ ├── utils_data.py
│ └── utils_local_dataset.py
│
├── requirements.txt
├── test_api.py
├── .gitignore
└── README.md
```

> Note: `models/`, `data/`, `venv/`, and `__pycache__/` are ignored via `.gitignore`.

---

## ⚙️ Installation

### 1. Clone the repo
```
git clone https://github.com/kspoojith/code-explainer.git

cd code-explainer
```


### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 🧠 Start the Backend (FastAPI)

Run:
```
uvicorn src.inference:app --reload --host 127.0.0.1 --port 8000
```


The backend loads:
- Fine-tuned CodeT5 model  
- Rule-based English → Telugu translator  

---

## 🖥️ Start the Frontend (Streamlit)

```
streamlit run src/streamlit_app.py
```

The Streamlit UI connects directly to the FastAPI backend.

---

## 📝 API Usage

### Endpoint
```
POST /explain
```

### Request Body
```json
{
  "code": "your python code here",
  "mode": "explain"
}
```
Sample Response
```
{
  "english": "English explanation here...",
  "telugu": "Telugu explanation here..."
}
```

🔧 How It Works
1. Preprocess Code

Cleanup and normalization.

2. English Explanation

Fine-tuned CodeT5 generates step-wise English code explanation.

3. Telugu Translation

A custom rule-based translator converts English lines into accurate, context-preserving Telugu.

Technical terms preserved

No mixed-language verbs

Very fast (no heavy model loading)

Test API:
```
python test_api.py
```

![Uploading SoExcited~GIF.gif…]()

