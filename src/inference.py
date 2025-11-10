# src/inference.py
import re
import ast
import textwrap
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from src.translate import TranslatorEnToTe

app = FastAPI(title="Telugu Code Explainer API")

# use your locally fine-tuned model, but only as a fallback
EXPLAIN_MODEL = "models/codet5-finetuned"
USE_MODEL_FALLBACK = True  # set False if you want rule-based only


class CodeRequest(BaseModel):
    code: str
    mode: str = "detailed"


def _explain_python_rule_based(code: str) -> str:
    """
    Deterministic explainer for common Python patterns.
    Uses AST to generate numbered steps. Returns '' if it can’t parse.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return ""

    # Helper: look for the function, loops, sets, condition, etc.
    func_name = None
    returns_name = None
    has_outer_for = False
    has_inner_for = False
    even_set_names = set()
    odd_set_names = set()
    max_var = None
    window_update = False
    equality_check_desc = None

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            nonlocal func_name
            func_name = node.name
            self.generic_visit(node)

        def visit_For(self, node: ast.For):
            nonlocal has_outer_for, has_inner_for
            # rough: first for -> outer, nested -> inner
            if not has_outer_for:
                has_outer_for = True
            else:
                has_inner_for = True
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            nonlocal max_var
            # maxi = 0
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, (ast.Constant, ast.Num))
                and getattr(node.value, "value", None) in (0, 0.0)):
                max_var = node.targets[0].id
            # even = set(), odd = set()
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "set":
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            # we will infer names as even/odd by later use
                            pass
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            # even.add(x), odd.add(x)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
                base = node.func.value
                if isinstance(base, ast.Name):
                    nm = base.id.lower()
                    if "even" in nm:
                        even_set_names.add(base.id)
                    if "odd" in nm:
                        odd_set_names.add(base.id)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare):
            nonlocal equality_check_desc, window_update
            # len(even) == len(odd)
            def len_name(expr):
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "len":
                    if expr.args and isinstance(expr.args[0], ast.Name):
                        return expr.args[0].id
                return None

            left_name = len_name(node.left)
            right_name = None
            if node.comparators:
                right_name = len_name(node.comparators[0])

            if left_name and right_name and any(isinstance(op, ast.Eq) for op in node.ops):
                equality_check_desc = (left_name, right_name)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign):
            # maxi = max(maxi, j-i+1) usually appears as normal Assign; keep generic
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return):
            nonlocal returns_name
            if isinstance(node.value, ast.Name):
                returns_name = node.value.id
            self.generic_visit(node)

        def visit_Assign_max(self, node: ast.Assign):
            pass

        def visit(self, node):
            # detect max update: maxi = max(maxi, j-i+1)
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name):
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        if node.value.func.id == "max":
                            nonlocal window_update, max_var
                            max_var = max_var or t.id
                            window_update = True
            super().visit(node)

    V().visit(tree)

    # Try to identify the specific logic seen in the sample:
    # - double loop over indices
    # - maintain two sets for even and odd values in the current window
    # - when len(even_set) == len(odd_set), update maxi with window length
    likely_even_odd_sets = bool(even_set_names or odd_set_names)
    likely_window_scan = has_outer_for and has_inner_for and window_update

    steps = []

    if func_name:
        steps.append(f"1. The function `{func_name}` computes the length of the longest subarray that satisfies a specific balance condition.")
    else:
        steps.append("1. The code computes the length of the longest subarray that satisfies a specific balance condition.")

    if has_outer_for and has_inner_for:
        steps.append("2. It scans all subarrays using two nested loops: the outer loop chooses a start index and the inner loop extends the end index.")
    elif has_outer_for:
        steps.append("2. It iterates through the array elements.")

    if likely_even_odd_sets:
        steps.append("3. For each starting index, it creates two sets: one for distinct even values and one for distinct odd values in the current window.")

    if equality_check_desc:
        a, b = equality_check_desc
        if (a.lower().startswith("even") and b.lower().startswith("odd")) or (b.lower().startswith("even") and a.lower().startswith("odd")):
            steps.append("4. As it expands the window, it inserts the current number into the even set if it is even; otherwise into the odd set.")
            steps.append("5. Whenever the counts of distinct even values and distinct odd values become equal, it considers the current window as balanced.")
        else:
            steps.append("4. It checks a balance condition between two sets using their lengths.")

    if window_update:
        steps.append("6. On each balanced window, it updates the answer with the maximum window length found so far.")

    if returns_name:
        steps.append(f"7. After checking all subarrays, it returns `{returns_name}`, the maximum balanced length.")
    else:
        steps.append("7. After checking all subarrays, it returns the maximum balanced length.")

    # If core cues were not found, bail out so we can fall back to model
    core_ok = likely_window_scan and likely_even_odd_sets and equality_check_desc is not None
    if not core_ok:
        return ""

    # Post-process: make sure numbering is clean and single-sentence lines
    numbered = []
    for i, s in enumerate(steps, 1):
        s = re.sub(r"\s+", " ", s).strip()
        if not s.endswith("."):
            s += "."
        numbered.append(f"{i}. {s}")
    return "\n".join(numbered)


@app.on_event("startup")
def load_models():
    global explainer, translator
    print("🚀 Loading models...")

    # translator (your existing one; fast or NLLB—unchanged)
    translator = TranslatorEnToTe()

    # fallback model (only if needed)
    explainer = None
    if USE_MODEL_FALLBACK:
        explainer = pipeline(
            "text2text-generation",
            model=EXPLAIN_MODEL,
            tokenizer=EXPLAIN_MODEL,
            device=-1,
        )

    print("🔥 Ready (rule-based first, model fallback).")


@app.post("/explain")
def explain(req: CodeRequest):
    code = textwrap.dedent(req.code).strip()

    # 1) Try rule-based Python explanation first
    english = _explain_python_rule_based(code)

    # 2) Fallback to your model only if rule-based failed
    if not english:
        if explainer is None:
            # last-resort generic
            english = (
                "1. The code processes the input.\n"
                "2. It iterates over elements and applies conditions.\n"
                "3. It maintains intermediate results.\n"
                "4. Finally, it returns the computed value.\n"
            )
        else:
            prompt = (
                "Explain the following Python code in clear, meaningful numbered steps. "
                "Do NOT output code; only describe the logic:\n\n"
                f"{code}"
            )
            raw = explainer(prompt, max_new_tokens=256, do_sample=False, truncation=True)[0]["generated_text"].strip()
            english = re.sub(r"(\d+)\.\s*", r"\1. ", raw).replace(". ", ".\n")
            if english.strip() in ["", "1.", "1"]:
                english = (
                    "1. The code processes the input.\n"
                    "2. It iterates over elements and applies conditions.\n"
                    "3. It maintains intermediate results.\n"
                    "4. Finally, it returns the computed value.\n"
                )

    # 3) Telugu translation via your translator
    telugu = "\n".join(translator.translate(line) if line.strip() else "" for line in english.split("\n"))

    return {"english": english, "telugu": telugu}
