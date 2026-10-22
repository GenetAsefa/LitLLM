"""
decode_candidates.py

Converts an exported candidate file (the *_candidates_tail.json or
*_candidates_head.json written by export_from_tuned_model.py), which holds
PyKEEN's integer IDs, back to the original Wikidata QIDs/PIDs used in the
LitWD train/valid/test files.

The mapping is rebuilt with the same loader (litwd_pykeen_loader.py) that
trained the model and produced the export, so the IDs line up. It is also
saved once as entity_to_id.json / relation_to_id.json next to the output, so
the mapping can be kept with the candidate files.

Usage:
    python decode_candidates.py \
        --dataset-dir ../LiterallyWikidata-main/Datasets/LitWD1K \
        --candidates runs/litwd1k/retriever_export/LitWD1K_rotate_tuned_candidates_tail.json \
        --out runs/litwd1k/retriever_export/LitWD1K_rotate_tuned_candidates_tail_qid.json

Output record (tail file): head, relation, true_tail, candidates, candidate_scores.
Head file: relation, tail, true_head, candidates, candidate_scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from litwd_pykeen_loader import load_litwd_triples_factories, save_id_mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    train_tf, _, _ = load_litwd_triples_factories(args.dataset_dir)
    id_to_entity = {i: label for label, i in train_tf.entity_to_id.items()}
    id_to_relation = {i: label for label, i in train_tf.relation_to_id.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_id_mappings(train_tf, args.out.parent)

    with open(args.candidates) as f:
        records = json.load(f)

    decoded = []
    for rec in records:
        new = {}
        for key, value in rec.items():
            if key == "relation_id":
                new["relation"] = id_to_relation[value]
            elif key == "candidate_ids":
                new["candidates"] = [id_to_entity[i] for i in value]
            elif key.endswith("_id"):
                new[key[: -len("_id")]] = id_to_entity[value]
            else:
                new[key] = value
        decoded.append(new)

    with open(args.out, "w") as f:
        json.dump(decoded, f)
    print(f"Decoded {len(decoded)} records -> {args.out}")
    print("Example:", json.dumps({k: (v[:3] if isinstance(v, list) else v) for k, v in decoded[0].items()}))


if __name__ == "__main__":
    main()