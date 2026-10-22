"""
litwd_pykeen_loader.py

Loads the LiterallyWikidata benchmark (LitWD19K / LitWD48K) structural
triples into PyKEEN TriplesFactory objects with a shared entity/relation
vocabulary across the train/valid/test splits.

LiterallyWikidata is NOT in PyKEEN's built-in dataset catalog (confirmed
absent as of PyKEEN 1.11.x), so this loader points PyKEEN's generic
TriplesFactory machinery at the raw tab-separated triple files from
https://github.com/GenetAsefa/LiterallyWikidata instead.

Usage:
    from litwd_pykeen_loader import load_litwd_triples_factories

    train_tf, valid_tf, test_tf = load_litwd_triples_factories(
        "/path/to/LiterallyWikidata/Datasets/LitWD19K"
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from pykeen.triples import TriplesFactory

DEFAULT_TRAIN_FILENAME = "train.txt"
DEFAULT_VALID_FILENAME = "valid.txt"
DEFAULT_TEST_FILENAME = "test.txt"


def load_litwd_triples_factories(
    dataset_dir: str | Path,
    train_filename: str = DEFAULT_TRAIN_FILENAME,
    valid_filename: str = DEFAULT_VALID_FILENAME,
    test_filename: str = DEFAULT_TEST_FILENAME,
    sep: str = "\t",
) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    """Load LitWD19K/LitWD48K structural triples as PyKEEN TriplesFactory objects.

    The three splits share one entity_to_id / relation_to_id vocabulary,
    built from the TRAIN split and then reused for valid/test — this is
    required so that entity/relation IDs are consistent across splits
    (PyKEEN will raise/warn on unseen labels otherwise, since LitWD is
    fully transductive: every valid/test entity is expected to already
    appear in train).

    Returns:
        (train_tf, valid_tf, test_tf)
    """
    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / train_filename
    valid_path = dataset_dir / valid_filename
    test_path = dataset_dir / test_filename

    for p in (train_path, valid_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected split file not found: {p}\n"
                "Double-check the filenames inside your LiterallyWikidata "
                "checkout against DEFAULT_*_FILENAME in this module, or "
                "pass the correct filenames explicitly."
            )

    train_tf = TriplesFactory.from_path(str(train_path), delimiter=sep)
    valid_tf = TriplesFactory.from_path(
        str(valid_path),
        delimiter=sep,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    test_tf = TriplesFactory.from_path(
        str(test_path),
        delimiter=sep,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    return train_tf, valid_tf, test_tf


def save_id_mappings(train_tf: TriplesFactory, out_dir: str | Path) -> None:
    """Dump entity_to_id / relation_to_id as JSON, so downstream code (KICGPT
    adapters, LiterallyLLM's context builder) can map PyKEEN's integer IDs
    back to the original Wikidata QIDs/PIDs used elsewhere in the pipeline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "entity_to_id.json", "w") as f:
        json.dump(train_tf.entity_to_id, f)
    with open(out_dir / "relation_to_id.json", "w") as f:
        json.dump(train_tf.relation_to_id, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check a LitWD dataset load.")
    parser.add_argument("dataset_dir", help="Path to e.g. .../LiterallyWikidata/Datasets/LitWD19K")
    args = parser.parse_args()

    train_tf, valid_tf, test_tf = load_litwd_triples_factories(args.dataset_dir)
    print(f"Entities: {train_tf.num_entities}")
    print(f"Relations: {train_tf.num_relations}")
    print(f"Train triples: {train_tf.num_triples}")
    print(f"Valid triples: {valid_tf.num_triples}")
    print(f"Test triples: {test_tf.num_triples}")




    