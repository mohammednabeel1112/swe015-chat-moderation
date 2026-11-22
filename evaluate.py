"""Evaluate the shipped inference paths against the same official test rows."""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from data import MODELS, ROOT, split
from moderation import LATENCY_BUDGET_MS, NAMES, Predictor


def cpu_name():
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def evaluate(models, output):
    output.mkdir(parents=True, exist_ok=True)
    texts, labels = split("test")
    rng = np.random.default_rng(42)
    timed_rows = rng.choice(len(texts), size=200, replace=False).tolist()
    test_rows = [{"test_row": i + 1, "reference": y} for i, y in enumerate(labels)]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {
        "dataset": "TweetEval offensive / OLID",
        "test_n": len(labels),
        "class_counts": np.bincount(labels).tolist(),
        "latency": {
            "unit": "ms",
            "warmups": 20,
            "runs": len(timed_rows),
            "threads": 1,
            "batch_size": 1,
            "budget_ms": LATENCY_BUDGET_MS,
            "hardware": cpu_name(),
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "sklearn": sklearn.__version__,
            "scope": "normalization through label, confidence and mask; excludes HTTP and UI",
        },
        "models": {},
    }
    timing_table = []
    with threadpool_limits(limits=1):
        for key in ("rules", "tfidf", "neural", "bert", "int8"):
            predictor = Predictor(key, models)
            scores = [predictor.classify(t) for t in texts]
            pred = [p["label"] for p in scores]
            for row, p in zip(test_rows, scores):
                row[key + "_label"] = p["label"]
                row[key + "_confidence"] = round(p["confidence"], 8)
            for i in range(20):
                predictor.classify(texts[i])
            times = [predictor.classify(texts[idx])["ms"] for idx in timed_rows]
            timing_table.extend(
                {"model": key, "test_row": idx + 1, "latency_ms": ms}
                for idx, ms in zip(timed_rows, times)
            )
            size = sum(p.stat().st_size for p in (models / key).rglob("*") if p.is_file())
            report["models"][key] = {
                "name": NAMES[key],
                "macro_f1": float(f1_score(labels, pred, average="macro")),
                "class_f1": f1_score(labels, pred, average=None, labels=[0, 1]).tolist(),
                "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
                "bytes": size,
                "size_mb": size / 1_000_000,
                "mean_ms": float(np.mean(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "max_ms": float(max(times)),
                "within_budget": int(np.sum(np.array(times) <= LATENCY_BUDGET_MS)),
            }
            print(key, report["models"][key], flush=True)
    for filename, rows in [("predictions.csv", test_rows), ("latency.csv", timing_table)]:
        with (output / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report["bert_int8_label_changes"] = sum(r["bert_label"] != r["int8_label"] for r in test_rows)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODELS)
    p.add_argument("--out", type=Path, default=ROOT / "reports")
    args = p.parse_args()
    evaluate(args.models, args.out)
