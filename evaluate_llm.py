#!/usr/bin/env python3
"""
evaluate_llm.py
A small utility to compare model predictions to references using multiple matching strategies.

Usage examples:
  # From a jsonl file (each line: {"prediction": "...", "reference": "..."})
  python evaluate_llm.py --file data.jsonl --mode fuzzy --threshold 0.85

  # From two plain text files (one example per line)
  python evaluate_llm.py --preds preds.txt --refs refs.txt --mode exact
"""
import argparse
import json
import re
import difflib
from typing import List, Dict
from statistics import mean


def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    return s


def exact_match(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def fuzzy_score(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def token_overlap(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def evaluate(preds: List[str], refs: List[str], mode: str = "exact", threshold: float = 0.9) -> Dict:
    assert len(preds) == len(refs), "preds and refs must have same length"
    N = len(preds)
    results = []
    matches = 0
    scores = []
    for p, r in zip(preds, refs):
        if mode == "exact":
            ok = exact_match(p, r)
            score = 1.0 if ok else 0.0
        elif mode == "fuzzy":
            score = fuzzy_score(p, r)
            ok = score >= threshold
        elif mode == "token":
            score = token_overlap(p, r)
            ok = score >= threshold
        else:
            raise ValueError("mode must be one of exact|fuzzy|token")
        matches += 1 if ok else 0
        scores.append(score)
        results.append({"pred": p, "ref": r, "score": score, "match": ok})
    summary = {
        "total": N,
        "matches": matches,
        "accuracy": matches / N if N else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "per_sample": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="jsonl with fields prediction/reference (or pred/ref)")
    parser.add_argument("--preds", help="file with predictions, one per line")
    parser.add_argument("--refs", help="file with references, one per line")
    parser.add_argument("--mode", choices=["exact", "fuzzy", "token"], default="exact")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for fuzzy/token modes")
    args = parser.parse_args()

    preds, refs = [], []
    if args.file:
        items = read_jsonl(args.file)
        for it in items:
            p = it.get("prediction") or it.get("pred") or it.get("output") or it.get("response") or it.get("result")
            r = it.get("reference") or it.get("ref") or it.get("label") or it.get("target")
            preds.append(p or "")
            refs.append(r or "")
    else:
        if not (args.preds and args.refs):
            parser.error("Either --file or both --preds and --refs must be provided")
        preds = read_lines(args.preds)
        refs = read_lines(args.refs)

    if len(preds) != len(refs):
        raise SystemExit("Lengths of predictions and references differ")

    summary = evaluate(preds, refs, mode=args.mode, threshold=args.threshold)

    # Print a compact report
    print("MODE:", args.mode, "THRESHOLD:", args.threshold)
    print(f"Total: {summary['total']}, Matches: {summary['matches']}, Accuracy: {summary['accuracy']:.4f}, Avg score: {summary['avg_score']:.4f}")
    print("\nSample results (first 10):")
    for i, s in enumerate(summary["per_sample"][:10], 1):
        print(f"{i:02d}. match={s['match']}, score={s['score']:.3f}, pred={s['pred']!r}, ref={s['ref']!r}")


if __name__ == "__main__":
    main()
