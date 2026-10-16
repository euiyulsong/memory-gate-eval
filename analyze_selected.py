import argparse
import json
from pathlib import Path


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(name, selected, all_rows):
    tp = sum(row["label"] == 1 for row in selected)
    fp = sum(row["label"] == 0 for row in selected)
    fn = sum(row["label"] == 1 for row in all_rows) - tp
    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else 0.0
    positives = sum(row["label"] == 1 for row in all_rows)
    recall = tp / positives if positives else 0.0
    return {
        "name": name,
        "selected_count": predicted_positive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure a post-filter over cases the gate predicted as search-needed."
    )
    parser.add_argument("--predictions", default="output/predictions.jsonl")
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    gate_selected = [row for row in rows if row.get("prediction") == 1]
    # A perfect selector is an oracle upper bound, not an implementable result.
    oracle_selected = [row for row in gate_selected if row["label"] == 1]

    # The current benchmark stores only a binary gate response. It has no
    # per-item confidence, candidate memory IDs, or post-filter predictions.
    report = {
        "n": len(rows),
        "gate_predicted_search": summarize("gate", gate_selected, rows),
        "oracle_selector_upper_bound": summarize("oracle_upper_bound", oracle_selected, rows),
        "selector_measurement": "not_available",
        "reason": (
            "Existing predictions contain a binary search/no-search decision only; "
            "they do not contain a selector output, candidate IDs, or evidence labels. "
            "An actual selector cannot be evaluated from this file without defining "
            "candidate memories and ground-truth relevant IDs."
        ),
        "requested_precision_threshold": args.threshold,
        "threshold_note": (
            "A precision target is not measurable from the existing run. The oracle "
            "upper bound assumes all false-positive gate cases can be perfectly removed."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
