# Math Reasoning Eval — Claude models, cost vs. accuracy

A small, rigorous **evaluation harness** that benchmarks Claude models on a curated set of
math problems and reports the **cost/accuracy trade-off**. Grading is **symbolic** (not string
matching), so answers are judged for mathematical equivalence.

Built by **César Daniel Reinoso** — mathematician (ESPOCH), MSc in Statistics with a
Data Science & AI track (in progress).

## Why this exists

Choosing a model is an economic decision, not just a quality one. This harness measures, on the
same task, how far accuracy climbs as you move Haiku → Sonnet → Opus, and what each point of
accuracy costs — the kind of evidence an AI team needs before committing a model to production.

## What it demonstrates

- **Eval design**: a clean benchmark with difficulty tiers (easy/medium/hard) and topic tags
  (algebra, calculus, probability, number theory), no train/test leakage.
- **Rigorous grading**: a symbolic grader (SymPy) that accepts any equivalent form —
  `x + 3` = `3 + x`, `3/8` = `0.375`, `pi/2`, `\boxed{...}`, `x = 4` — instead of brittle
  string equality. 10/10 grader unit tests; every canonical answer independently verified by CAS.
- **Cost accounting**: real USD cost per model from token usage, with per-problem and total cost.
- **Domain expertise**: the problem set and answer keys are authored and CAS-verified by a
  mathematician. The questions are in Spanish — the grader is language-agnostic, so the harness
  doubles as a **multilingual** reasoning eval.
- **Engineering**: async concurrency, graceful error/refusal handling, reproducible JSONL output.

## Results

Run `python run_eval.py` and the report lands in `results/reporte.md` (summary table by model,
accuracy by difficulty, and total run cost). Paste the headline table here after your first run.

| Model | Accuracy | Total cost | Cost/problem |
|---|---|---|---|
| claude-haiku-5-5 | _run it_ | | |
| claude-sonnet-5-5 | | | |
| claude-opus-5-5 | | | |

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."

python grader.py                          # grader self-test (no API, free)
python run_eval.py --limit 2              # cheap smoke test (2 problems)
python run_eval.py                        # full run: 3 models x 20 problems
```

Useful flags: `--models claude-sonnet-5-5 claude-opus-5-5`, `--repeats 3` (reliability),
`--no-thinking`, `--concurrency 8`, `--limit N`.

## Files

| File | Purpose |
|---|---|
| `problems.jsonl` | 20 problems: id, topic, difficulty, statement, canonical answer |
| `grader.py` | final-answer extraction + symbolic equivalence (SymPy) + self-tests |
| `run_eval.py` | async runner: calls the Messages API, grades, tracks cost, writes the report |
| `results/` | generated: `reporte.md` + `resultados.jsonl` (gitignored) |

## Notes

- Models use adaptive thinking by default; thinking tokens are billed as output and included in cost.
- Prices are first-party Anthropic API rates (Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10,
  Opus 5.5 $4/$20 per 1M tokens).
- The API key is read from the environment and never stored in the repo.
