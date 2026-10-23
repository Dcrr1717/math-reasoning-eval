"""Math reasoning benchmark: compares Claude models on cost vs. accuracy.

For each (model, problem) it makes one Messages API call, asks the model to reason and
finish with 'FINAL: <answer>', grades with the symbolic grader (grader.py) and
accumulates tokens -> cost in dollars. Produces a markdown report and a JSONL of
raw results.

Usage:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...        (or set on Windows)
    python run_eval.py --limit 2               # cheap test (2 problems)
    python run_eval.py                          # full run (3 models x 20 problems)
    python run_eval.py --models claude-sonnet-5-5 claude-opus-5-5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from anthropic import AsyncAnthropic

from grader import calificar

# USD rates per million tokens (first-party Anthropic API). Prompts < 100K.
PRECIOS = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
MODELOS_POR_DEFECTO = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

SYSTEM = (
    "You are a rigorous math assistant. Solve the problem showing the "
    "reasoning needed and, on the LAST line, write exactly 'FINAL: <answer>', "
    "with the answer in its simplest form (a number, a fraction like 3/8, or an "
    "expression). Do not include units or extra text on that line."
)


def costo(modelo: str, tok_in: int, tok_out: int) -> float:
    pin, pout = PRECIOS.get(modelo, (0.0, 0.0))
    return tok_in / 1e6 * pin + tok_out / 1e6 * pout


def cargar_problemas(path: str, limite: int | None) -> list[dict]:
    probs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return probs[:limite] if limite else probs


async def una_llamada(client, sem, modelo, prob, repeticion, max_tokens, pensar):
    async with sem:
        t0 = time.time()
        kwargs = dict(model=modelo, max_tokens=max_tokens, system=SYSTEM,
                      messages=[{"role": "user", "content": prob["problem"]}])
        if pensar:
            kwargs["thinking"] = {"type": "adaptive"}
        try:
            resp = await client.messages.create(**kwargs)
        except Exception as e:
            return {"id": prob["id"], "modelo": modelo, "rep": repeticion, "error": str(e),
                    "correcto": False, "costo": 0.0, "tok_in": 0, "tok_out": 0, "seg": 0.0}
        seg = time.time() - t0
        if resp.stop_reason == "refusal":
            texto = ""
        else:
            texto = "".join(b.text for b in resp.content if b.type == "text")
        correcto, extraido = calificar(texto, prob["answer"])
        ti, to = resp.usage.input_tokens, resp.usage.output_tokens
        return {"id": prob["id"], "topic": prob["topic"], "difficulty": prob["difficulty"],
                "modelo": modelo, "rep": repeticion, "esperado": prob["answer"],
                "extraido": extraido, "correcto": correcto, "truncado": resp.stop_reason == "max_tokens",
                "tok_in": ti, "tok_out": to, "costo": costo(modelo, ti, to), "seg": round(seg, 2)}


async def correr(args):
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    problemas = cargar_problemas(args.problems, args.limit)
    tareas = [una_llamada(client, sem, m, p, r, args.max_tokens, not args.no_thinking)
              for m in args.models for p in problemas for r in range(args.repeats)]
    print(f"Running {len(tareas)} calls: {len(args.models)} models x {len(problemas)} "
          f"problems x {args.repeats} repeat(s)...")
    resultados = await asyncio.gather(*tareas)
    return problemas, resultados


def reporte(problemas, resultados, args) -> str:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultados.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in resultados), encoding="utf-8")

    por_mod = defaultdict(list)
    for r in resultados:
        por_mod[r["modelo"]].append(r)
    dificultades = ["easy", "medium", "hard"]

    L = ["# Math reasoning benchmark - Claude\n",
         f"{len(problemas)} problems | {args.repeats} repeat(s) | "
         f"thinking: {'off' if args.no_thinking else 'adaptive'}\n",
         "\n## Summary by model\n",
         "| Model | Accuracy | Total cost | Cost/problem | Tokens (in/out) | Mean lat. |",
         "|---|---|---|---|---|---|"]
    filas_costo = {}
    for m in args.models:
        rs = por_mod[m]
        n = len(rs)
        acc = sum(x["correcto"] for x in rs) / n if n else 0
        ct = sum(x["costo"] for x in rs)
        ti = sum(x["tok_in"] for x in rs)
        to = sum(x["tok_out"] for x in rs)
        lat = sum(x["seg"] for x in rs) / n if n else 0
        filas_costo[m] = (acc, ct)
        L.append(f"| {m} | {acc*100:.0f}% | ${ct:.4f} | ${ct/max(len(problemas)*args.repeats,1):.4f} "
                 f"| {ti:,}/{to:,} | {lat:.1f}s |")

    L.append("\n## Accuracy by difficulty\n")
    L.append("| Modelo | " + " | ".join(dificultades) + " |")
    L.append("|---|" + "|".join(["---"] * len(dificultades)) + "|")
    for m in args.models:
        rs = por_mod[m]
        celdas = []
        for d in dificultades:
            sub = [x for x in rs if x.get("difficulty") == d]
            celdas.append(f"{sum(x['correcto'] for x in sub)/len(sub)*100:.0f}%" if sub else "-")
        L.append(f"| {m} | " + " | ".join(celdas) + " |")

    costo_total = sum(x["costo"] for x in resultados)
    errores = [x for x in resultados if x.get("error")]
    L.append(f"\n**Total cost of this run: ${costo_total:.4f}**")
    if errores:
        L.append(f"\n> {len(errores)} call(s) with errors (e.g. {errores[0]['error'][:120]})")
    L.append("\n## Reading\n"
             "The cost-accuracy frontier: the chosen model should sit at the point where a gain "
             "in accuracy no longer justifies the jump in cost. Haiku gives the cheap floor; Opus, the ceiling.\n")
    texto = "\n".join(L)
    (out / "reporte.md").write_text(texto, encoding="utf-8")
    return texto, costo_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELOS_POR_DEFECTO)
    ap.add_argument("--problems", default="problems.jsonl")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N problems")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment before running.")
        raise SystemExit(1)

    problemas, resultados = asyncio.run(correr(args))
    texto, costo_total = reporte(problemas, resultados, args)
    print("\n" + texto)
    print(f"\nSaved to {args.out}/reporte.md and {args.out}/resultados.jsonl")


if __name__ == "__main__":
    main()
