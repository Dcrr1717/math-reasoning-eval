"""Calificador simbolico para el benchmark de razonamiento matematico.

Extrae la respuesta final del texto del modelo (linea 'FINAL: ...') y la compara
con la respuesta canonica usando equivalencia simbolica (sympy), no coincidencia
exacta de cadenas. Asi 'x + 3', '3+x' y '3 + x' cuentan como correctas, y '3/8'
equivale a '0.375'.
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
    convert_xor,  # permite usar ^ como potencia
)

# Normalizaciones de simbolos que suelen escribir los modelos
_SUBS = {
    "\u03c0": "pi",   # pi
    "\u221a": "sqrt", # raiz
    "\u00f7": "/",
    "\u00d7": "*",
    "\u2212": "-",    # signo menos unicode
    ",": "",          # separador de miles (1,024 -> 1024); los decimales usan punto
}


def extract_final(texto: str) -> str | None:
    """Toma lo que sigue a la ultima aparicion de 'FINAL:' (o 'RESPUESTA:')."""
    if not texto:
        return None
    patrones = re.findall(r"(?:FINAL|RESPUESTA)\s*[:=]\s*(.+)", texto, flags=re.IGNORECASE)
    if patrones:
        bruto = patrones[-1].strip()
    else:
        # respaldo: ultima linea no vacia
        lineas = [l.strip() for l in texto.strip().splitlines() if l.strip()]
        if not lineas:
            return None
        bruto = lineas[-1]
    # limpiar envoltorios comunes
    bruto = bruto.strip().strip("$").strip()
    bruto = re.sub(r"\\boxed\{(.+?)\}", r"\1", bruto)   # \boxed{...}
    bruto = re.sub(r"\\text\{.*?\}", "", bruto)
    bruto = bruto.rstrip(".")
    # si viene 'x = 4' tomamos lo de la derecha del ultimo '='
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
    """True si ambas expresiones son matematicamente equivalentes."""
    if respuesta_modelo is None:
        return False
    try:
        a = _a_expr(respuesta_modelo)
        b = _a_expr(respuesta_canonica)
    except Exception:
        # ultimo recurso: comparacion textual normalizada
        return respuesta_modelo.replace(" ", "").lower() == respuesta_canonica.replace(" ", "").lower()
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        # comparacion numerica para constantes
        if abs(complex(sp.N(a)) - complex(sp.N(b))) < 1e-9:
            return True
    except Exception:
        pass
    return False


def calificar(texto_modelo: str, respuesta_canonica: str) -> tuple[bool, str | None]:
    final = extract_final(texto_modelo)
    return son_equivalentes(final, respuesta_canonica), final


if __name__ == "__main__":
    # autocomprobacion rapida (sin API)
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
        marca = "OK" if res == esperado else "FALLA"
        if res == esperado:
            ok += 1
        print(f"[{marca}] canon={canon!r} extraido={final!r} -> {res} (esperado {esperado})")
    print(f"\n{ok}/{len(casos)} casos de prueba del calificador pasaron")
