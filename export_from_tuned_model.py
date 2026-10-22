"""
export_from_tuned_model.py

Takes an already-trained PyKEEN model -- e.g. one of the trained_model.pkl
files produced by tune_retriever.py's replicate step -- and produces the
same two artifacts train_base_retriever.py produces for a freshly-trained
model: the filtered top-k candidate export per test triple, and Recall@k
computed from it. 

Usage:
    python export_from_tuned_model.py \
        --dataset-dir /path/to/LiterallyWikidata/Datasets/LitWD19K \
        --dataset-name litwd19k \
        --model-name rotate_tuned \
        --trained-model-path ./hpo_outputs/litwd19k_rotate/litwd19k_rotate_hpo_best_replicates/replicates/replicate-00000/trained_model.pkl \
        --k 50 \
        --out-dir ./retriever_outputs/litwd19k_rotate_tuned \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from litwd_pykeen_loader import load_litwd_triples_factories
from train_base_retriever import (
    build_known_triples_index,
    compute_recall_at_k,
    export_filtered_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", required=True, help="Label for output filenames, e.g. rotate_tuned")
    parser.add_argument("--trained-model-path", required=True, help="Path to a trained_model.pkl saved by tune_retriever.py's replicate step")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[1, 3, 10, 20])
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if any(rk > args.k for rk in args.recall_ks):
        raise ValueError(
            f"--recall-ks contains a value larger than --k ({args.k}): "
            f"{[rk for rk in args.recall_ks if rk > args.k]}. Raise --k or drop those values "
            "(same rule as train_base_retriever.py, for the same reason: Recall@k beyond the "
            "exported candidate-set size is not meaningful)."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, valid_tf, test_tf = load_litwd_triples_factories(args.dataset_dir)

    model = torch.load(args.trained_model_path, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()

    # Sanity check: the loaded model's entity/relation counts must match
    # this dataset, or the export below would silently score against the
    # wrong vocabulary and produce garbage candidate IDs.
    if model.num_entities != train_tf.num_entities or model.num_relations != train_tf.num_relations:
        raise ValueError(
            f"Loaded model has {model.num_entities} entities / {model.num_relations} relations, "
            f"but {args.dataset_dir} has {train_tf.num_entities} / {train_tf.num_relations}. "
            "Are --dataset-dir and --trained-model-path pointing at the same dataset?"
        )

    hr_to_tails, rt_to_heads = build_known_triples_index(train_tf, valid_tf, test_tf)
    tail_records, head_records = export_filtered_candidates(
        model, test_tf, hr_to_tails, rt_to_heads, k=args.k, device=device, batch_size=args.eval_batch_size
    )

    with open(out_dir / f"{args.dataset_name}_{args.model_name}_candidates_tail.json", "w") as f:
        json.dump(tail_records, f)
    with open(out_dir / f"{args.dataset_name}_{args.model_name}_candidates_head.json", "w") as f:
        json.dump(head_records, f)

    recall_ks = sorted(set(args.recall_ks + [args.k]))
    tail_recall = compute_recall_at_k(tail_records, "true_tail_id", recall_ks)
    head_recall = compute_recall_at_k(head_records, "true_head_id", recall_ks)
    combined_recall = {
        k: (tail_recall[k] * len(tail_records) + head_recall[k] * len(head_records))
        / (len(tail_records) + len(head_records))
        for k in recall_ks
    }
    recall_report = {
        "tail_prediction": tail_recall,
        "head_prediction": head_recall,
        "combined": combined_recall,
        "num_test_triples": test_tf.num_triples,
        "exported_k": args.k,
        "source_model": args.trained_model_path,
    }
    with open(out_dir / f"{args.dataset_name}_{args.model_name}_recall_at_k.json", "w") as f:
        json.dump(recall_report, f, indent=2)

    print(json.dumps(recall_report, indent=2))
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()