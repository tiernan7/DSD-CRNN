from pathlib import Path
import StrikePy
import importlib

import sympy as sym
from pathlib import Path
import StrikePy
import importlib
from math import inf
import io
import sys
from contextlib import redirect_stdout
from StrikePy.strike_goldd import strike_goldd

def extract_summary(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "SUMMARY" in line:
            return "\n".join(lines[i:])
    return ""  # SUMMARY not found


def strike_summary(name):
    _, text = run_strikepy_capture(strike_goldd, name)
    print(extract_summary(text))


def run_strikepy_capture(fn, *args, **kwargs):
    """
    Run a function while capturing stdout.
    Returns (result, full_text_output).
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()

def write_strikepy_model(modelname: str, model: dict) -> Path:
    models_dir = Path(StrikePy.__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Collect symbol names from p and x so we can define them in the generated module
    def is_sym(obj): return isinstance(obj, sym.Symbol)
    def sym_name(obj): return obj.name

    names = []
    for block in (model["p"], model["x"]):
        for row in block:
            for obj in row:
                if is_sym(obj):
                    names.append(sym_name(obj))
    # unique, preserve order
    names = list(dict.fromkeys(names))

    def serialize(obj):
        return obj.name if is_sym(obj) else repr(obj)

    def serialize_block(block):
        return "[" + ", ".join(
            "[" + ", ".join(serialize(obj) for obj in row) + "]"
            for row in block
        ) + "]"

    src = (
        "# Auto-generated StrikePy model\n"
        "from sympy import symbols\n\n"
        + (f"{', '.join(names)} = symbols('{ ' '.join(names) }')\n\n" if names else "")
        + f"p = {serialize_block(model['p'])}\n"
        + f"x = {serialize_block(model['x'])}\n"
        + f"u = {repr(model.get('u', []))}\n"  # <-- REQUIRED by your StrikePy driver
        + f"w = {repr(model.get('w', []))}\n"
        + f"f = {serialize_block(model['f'])}\n"
        + f"h = {serialize_block(model['h'])}\n"
    )

    path = models_dir / f"{modelname}.py"
    path.write_text(src)
    importlib.invalidate_caches()
    return path

def write_strikepy_options(
    optname: str,
    modelname: str,
    *,
    checkObser: int = 1,
    maxLietime = inf,
    nnzDerU = None,
    nnzDerW = None,
    prev_ident_pars = None,
) -> Path:
    custom_dir = Path(StrikePy.__file__).resolve().parent / "custom_options"
    custom_dir.mkdir(exist_ok=True)

    if nnzDerU is None: nnzDerU = [inf]   # matches your template
    if nnzDerW is None: nnzDerW = [inf]
    if prev_ident_pars is None: prev_ident_pars = []

    src = (
        "# Auto-generated StrikePy options\n"
        "import sympy as sym\n"
        "from math import inf\n\n"
        f"modelname = '{modelname}'\n"
        f"checkObser = {int(checkObser)}\n"
        f"maxLietime = {('inf' if maxLietime == inf else repr(maxLietime))}\n"
        f"nnzDerU = {repr(nnzDerU)}\n"
        f"nnzDerW = {repr(nnzDerW)}\n"
        f"prev_ident_pars = {repr(prev_ident_pars)}\n"
    )

    path = custom_dir / f"{optname}.py"
    path.write_text(src)
    importlib.invalidate_caches()
    return path

