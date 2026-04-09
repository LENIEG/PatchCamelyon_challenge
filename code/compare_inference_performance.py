"""
Compare two inference outputs produced by model inference scripts.

Expected CSV schema for both files:
- sample_index
- y_true
- y_pred
- logit_class_0
- logit_class_1
- prob_class_0
- prob_class_1

Outputs:
- JSON summary with per-model and head-to-head comparison metrics
- CSV with per-sample disagreement analysis

Example usage:
python code/compare_inference_performance.py --model-a-predictions ./results/convnext_tiny/test_predictions.csv --model-b-predictions ./results/vit_small_dino/test_predictions.csv --model-a-name ConvNeXt-Tiny --model-b-name ViT-Small-DINO --output-prefix convnext_vs_vit
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def load_predictions(csv_path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "sample_index",
            "y_true",
            "y_pred",
            "logit_class_0",
            "logit_class_1",
            "prob_class_0",
            "prob_class_1",
        }
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "logit_class_0": float(row["logit_class_0"]),
                    "logit_class_1": float(row["logit_class_1"]),
                    "prob_class_0": float(row["prob_class_0"]),
                    "prob_class_1": float(row["prob_class_1"]),
                }
            )

    rows.sort(key=lambda x: x["sample_index"])
    return rows


def compute_binary_metrics(rows):
    tp = tn = fp = fn = 0
    for r in rows:
        y_true = r["y_true"]
        y_pred = r["y_pred"]
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    balanced_acc = 0.5 * (recall + specificity)

    return {
        "num_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compare_rows(rows_a, rows_b):
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"Prediction files have different lengths: {len(rows_a)} vs {len(rows_b)}"
        )

    disagreements = []
    both_correct = both_wrong = a_only_correct = b_only_correct = 0

    for a, b in zip(rows_a, rows_b):
        if a["sample_index"] != b["sample_index"]:
            raise ValueError(
                f"sample_index mismatch: {a['sample_index']} vs {b['sample_index']}"
            )
        if a["y_true"] != b["y_true"]:
            raise ValueError(
                f"y_true mismatch at sample {a['sample_index']}: {a['y_true']} vs {b['y_true']}"
            )

        y_true = a["y_true"]
        a_correct = int(a["y_pred"] == y_true)
        b_correct = int(b["y_pred"] == y_true)

        if a_correct and b_correct:
            both_correct += 1
        elif (not a_correct) and (not b_correct):
            both_wrong += 1
        elif a_correct and (not b_correct):
            a_only_correct += 1
        else:
            b_only_correct += 1

        disagree = int(a["y_pred"] != b["y_pred"])
        if disagree:
            disagreements.append(
                {
                    "sample_index": a["sample_index"],
                    "y_true": y_true,
                    "model_a_pred": a["y_pred"],
                    "model_b_pred": b["y_pred"],
                    "model_a_prob_class_1": a["prob_class_1"],
                    "model_b_prob_class_1": b["prob_class_1"],
                    "model_a_correct": a_correct,
                    "model_b_correct": b_correct,
                }
            )

    total = len(rows_a)
    agreement_count = total - len(disagreements)
    agreement_rate = safe_div(agreement_count, total)

    # McNemar's chi-square (continuity corrected), useful for paired classifier comparison.
    b = a_only_correct
    c = b_only_correct
    mcnemar_chi2 = safe_div((abs(b - c) - 1.0) ** 2, b + c) if (b + c) > 0 else 0.0

    return {
        "num_samples": total,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_rate,
        "disagreement_count": len(disagreements),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "model_a_only_correct": a_only_correct,
        "model_b_only_correct": b_only_correct,
        "mcnemar_chi2_continuity_corrected": mcnemar_chi2,
    }, disagreements


def load_summary_if_exists(summary_path):
    if not summary_path:
        return None
    path = Path(summary_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_disagreements_csv(path, disagreements):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "y_true",
                "model_a_pred",
                "model_b_pred",
                "model_a_prob_class_1",
                "model_b_prob_class_1",
                "model_a_correct",
                "model_b_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(disagreements)


def main():
    debug_config = {
        "model_a_name": "ConvNeXt-Tiny",
        "model_b_name": "ViT-Small-DINO",
        "model_a_predictions": "./results/convnext_tiny/test_predictions.csv",
        "model_b_predictions": "./results/vit_small_dino/test_predictions.csv",
        "model_a_summary": "./results/convnext_tiny/test_summary.json",
        "model_b_summary": "./results/vit_small_dino/test_summary.json",
        "output_dir": "./results/comparisons",
        "output_prefix": "convnext_vs_vit",
    }

    parser = argparse.ArgumentParser(description="Compare two inference prediction files")
    parser.add_argument("--model-a-name", type=str, default="ConvNeXt-Tiny")
    parser.add_argument("--model-b-name", type=str, default="ViT-Small-DINO")
    parser.add_argument("--model-a-predictions", type=str, required=False)
    parser.add_argument("--model-b-predictions", type=str, required=False)
    parser.add_argument("--model-a-summary", type=str, default="")
    parser.add_argument("--model-b-summary", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./results/comparisons")
    parser.add_argument("--output-prefix", type=str, default="convnext_vs_vit")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        config = debug_config
        print(">> Running in DEBUG mode (using debug defaults) <<")
    else:
        if not args.model_a_predictions or not args.model_b_predictions:
            raise ValueError(
                "Both --model-a-predictions and --model-b-predictions are required in CLI mode"
            )
        config = {
            "model_a_name": args.model_a_name,
            "model_b_name": args.model_b_name,
            "model_a_predictions": args.model_a_predictions,
            "model_b_predictions": args.model_b_predictions,
            "model_a_summary": args.model_a_summary,
            "model_b_summary": args.model_b_summary,
            "output_dir": args.output_dir,
            "output_prefix": args.output_prefix,
        }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_a = load_predictions(Path(config["model_a_predictions"]))
    rows_b = load_predictions(Path(config["model_b_predictions"]))

    metrics_a = compute_binary_metrics(rows_a)
    metrics_b = compute_binary_metrics(rows_b)

    head_to_head, disagreements = compare_rows(rows_a, rows_b)

    acc_delta = metrics_a["accuracy"] - metrics_b["accuracy"]
    if math.isclose(acc_delta, 0.0, abs_tol=1e-12):
        winner = "tie"
    else:
        winner = config["model_a_name"] if acc_delta > 0 else config["model_b_name"]

    comparison = {
        "model_a": {
            "name": config["model_a_name"],
            "predictions_path": str(Path(config["model_a_predictions"])),
            "summary_path": str(Path(config["model_a_summary"])) if config["model_a_summary"] else "",
            "metrics": metrics_a,
            "reported_summary": load_summary_if_exists(config["model_a_summary"]),
        },
        "model_b": {
            "name": config["model_b_name"],
            "predictions_path": str(Path(config["model_b_predictions"])),
            "summary_path": str(Path(config["model_b_summary"])) if config["model_b_summary"] else "",
            "metrics": metrics_b,
            "reported_summary": load_summary_if_exists(config["model_b_summary"]),
        },
        "head_to_head": head_to_head,
        "accuracy_delta_model_a_minus_model_b": acc_delta,
        "winner_by_accuracy": winner,
    }

    json_path = output_dir / f"{config['output_prefix']}_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    disagreement_csv = output_dir / f"{config['output_prefix']}_disagreements.csv"
    save_disagreements_csv(disagreement_csv, disagreements)

    print("\n" + "=" * 70)
    print("Comparison complete")
    print("=" * 70)
    print(f"Model A: {config['model_a_name']}")
    print(f"  Accuracy: {metrics_a['accuracy']:.6f} | F1: {metrics_a['f1']:.6f}")
    print(f"Model B: {config['model_b_name']}")
    print(f"  Accuracy: {metrics_b['accuracy']:.6f} | F1: {metrics_b['f1']:.6f}")
    print(f"Winner (accuracy): {winner}")
    print(f"Agreement rate: {head_to_head['agreement_rate']:.6f}")
    print(f"Disagreement count: {head_to_head['disagreement_count']}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved disagreement CSV: {disagreement_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
