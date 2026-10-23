"""Symbolic grader for the math reasoning benchmark.

Extracts the model's final answer from its text (the 'FINAL: ...' line) and compares
it against the canonical answer using symbolic equivalence (sympy), not exact string
matching. This way 'x + 3', '3+x' and '3 + x' all count as correct, and '3/8' is
equivalent to '0.375'.
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,  # allows using ^ as exponentiation
)

# Symbol normalizations that models commonly write
_SUBS = {
    "\u03c0": "pi",   # pi
    "\u221a": "sqrt", # square root
    "\u00f7": "/",
    "\u00d7": "*",
    "\u2212": "-",    # unicode minus sign
    ",": "",          # thousands separator (1,024 -> 1024); decimals use a dot
}


def extract_final(texto: str) -> str | None:
    """Takes whatever follows the last occurrence of 'FINAL:' (or 'RESPUESTA:')."""
    if not texto:
        return None
    patrones = re.findall(r"(?:FINAL|RESPUESTA)\s*[:=]\s*(.+)", texto, flags=re.IGNORECASE)
    if patrones:
        bruto = patrones[-1].strip()
    else:
        # fallback: last non-empty line
        lineas = [l.strip() for l in texto.strip().splitlines() if l.strip()]
        if not lineas:
            return None
        bruto = lineas[-1]
    # strip common wrappers
    bruto = bruto.strip().strip("$").strip()
    bruto = re.sub(r"\\boxed\{(.+?)\}", r"\1", bruto)   # \boxed{...}
    bruto = re.sub(r"\\text\{.*?\}", "", bruto)
    bruto = bruto.rstrip(".")
    # if it comes as 'x = 4' we take what is to the right of the last '='
    if bruto.count("=") >= 1 and not bruto.lstrip().startswith("="):
        bruto = bruto.split("=")[-1].strip()
    for a, b in _SUBS.items():
        bruto = bruto.replace(a, b)
    return bruto.strip()


def _a_expr(s: str):
    s = s.strip()
    for a, b in _SUBS.items():
        s = s.replace(a, b)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)


def son_equivalentes(respuesta_modelo: str, respuesta_canonica: str) -> bool:
    """True if both expressions are mathematically equivalent."""
    if respuesta_modelo is None:
        return False
    try:
        a = _a_expr(respuesta_modelo)
        b = _a_expr(respuesta_canonica)
    except Exception:
        # last resort: normalized text comparison
        return respuesta_modelo.replace(" ", "").lower() == respuesta_canonica.replace(" ", "").lower()
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        # numeric comparison for constants
        if abs(complex(sp.N(a)) - complex(sp.N(b))) < 1e-9:
            return True
    except Exception:
        pass
    return False


def calificar(texto_modelo: str, respuesta_canonica: str) -> tuple[bool, str | None]:
    final = extract_final(texto_modelo)
    return son_equivalentes(final, respuesta_canonica), final


if __name__ == "__main__":
    # quick self-check (no API)
    casos = [
        ("razonamiento...\nFINAL: -12", "-12", True),
        ("FINAL: x + 3", "x + 3", True),
        ("FINAL: 3 + x", "x + 3", True),
        ("FINAL: 0.375", "3/8", True),
        ("FINAL: pi/2", "pi/2", True),
        ("La respuesta es \\boxed{4/3}\nFINAL: 4/3", "4/3", True),
        ("FINAL: 1,024", "1024", True),
        ("FINAL: x = 4", "4", True),
        ("FINAL: 7", "5", False),
        ("FINAL: 3*x**2", "3*x**2", True),
    ]
    ok = 0
    for texto, canon, esperado in casos:
        res, final = calificar(texto, canon)
        marca = "OK" if res == esperado else "FAIL"
        if res == esperado:
            ok += 1
        print(f"[{marca}] canon={canon!r} extracted={final!r} -> {res} (expected {esperado})")
    print(f"\n{ok}/{len(casos)} grader test cases passed")
