
"""
train_base_retriever.py
 
Usage example:
    python train_base_retriever.py \
        --dataset-dir /path/to/LiterallyWikidata/Datasets/LitWD19K \
        --dataset-name litwd19k \
        --model rotate \
        --k 20 \
        --epochs 200 \
        --embedding-dim 200 \
        --out-dir ./retriever_outputs/litwd19k_rotate
 
    python train_base_retriever.py \
        --dataset-dir /path/to/LiterallyWikidata/Datasets/LitWD48K \
        --dataset-name litwd48k \
        --model tucker \
        --k 20 \
        --epochs 200 \
        --embedding-dim 200 \
        --out-dir ./retriever_outputs/litwd48k_tucker
 
Requires: pip install pykeen torch pandas tqdm

"""
 
from __future__ import annotations
 
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
 
import torch
from tqdm import tqdm
 
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
 
from litwd_pykeen_loader import load_litwd_triples_factories, save_id_mappings
 
 
def build_known_triples_index(
    *triples_factories: TriplesFactory,
) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
    """Build the filtering index used for the standard *filtered* link-prediction
    protocol (Bordes et al., 2013): when scoring candidates for a given query,
    every OTHER known-true answer to that same (h, r) or (r, t) query — across
    train+valid+test combined — is excluded from the ranking before computing
    rank/top-k, except the one true answer for the query currently being scored.
 
    Returns:
        hr_to_tails: (head_id, relation_id) -> set of all known true tail_ids
        rt_to_heads: (relation_id, tail_id) -> set of all known true head_ids
    """
    hr_to_tails: Dict[Tuple[int, int], Set[int]] = {}
    rt_to_heads: Dict[Tuple[int, int], Set[int]] = {}
 
    for tf in triples_factories:
        mapped = tf.mapped_triples  # (num_triples, 3) LongTensor: [head, relation, tail]
        for h, r, t in mapped.tolist():
            hr_to_tails.setdefault((h, r), set()).add(t)
            rt_to_heads.setdefault((r, t), set()).add(h)
 
    return hr_to_tails, rt_to_heads
 
 
@torch.no_grad()
def export_filtered_candidates(
    model,
    test_tf: TriplesFactory,
    hr_to_tails: Dict[Tuple[int, int], Set[int]],
    rt_to_heads: Dict[Tuple[int, int], Set[int]],
    k: int,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[dict], List[dict]]:
    """For every test triple, compute the filtered top-k tail candidates (given
    h, r) and the filtered top-k head candidates (given r, t), using the
    model's raw scoring functions directly (model.score_t / model.score_h).
 
    This mirrors exactly what PyKEEN's own RankBasedEvaluator does internally
    to compute filtered MRR/Hits@k — the difference is we keep the actual
    top-k candidate IDs and scores instead of collapsing them into a single
    rank number, since that's what a downstream reranker (KICGPT,
    LiterallyLLM) needs as input.
 
    Returns:
        tail_records: one dict per test triple with the tail-prediction query,
            true answer, and ranked candidate list
        head_records: symmetric, for head-prediction queries
    """
    num_entities = model.num_entities
    mapped = test_tf.mapped_triples.to(device)
    n = mapped.shape[0]
 
    tail_records: List[dict] = []
    head_records: List[dict] = []
 
    for start in tqdm(range(0, n, batch_size), desc="Exporting filtered candidates"):
        batch = mapped[start : start + batch_size]
        heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
 
        # ---- tail prediction: (h, r) -> rank all tails ----
        hr_batch = torch.stack([heads, relations], dim=1)
        tail_scores = model.score_t(hr_batch).clone()  # (batch, num_entities)
        for i in range(batch.shape[0]):
            h, r, t = int(heads[i]), int(relations[i]), int(tails[i])
            known = hr_to_tails.get((h, r), set())
            if known:
                mask_idx = torch.tensor(
                    [e for e in known if e != t], dtype=torch.long, device=device
                )
                if mask_idx.numel() > 0:
                    tail_scores[i, mask_idx] = float("-inf")
            topk_scores, topk_idx = torch.topk(
                tail_scores[i], min(k, num_entities)
            )
            tail_records.append(
                {
                    "head_id": h,
                    "relation_id": r,
                    "true_tail_id": t,
                    "candidate_ids": topk_idx.cpu().tolist(),
                    "candidate_scores": topk_scores.cpu().tolist(),
                }
            )
 
        # ---- head prediction: (r, t) -> rank all heads ----
        rt_batch = torch.stack([relations, tails], dim=1)
        head_scores = model.score_h(rt_batch).clone()  # (batch, num_entities)
        for i in range(batch.shape[0]):
            h, r, t = int(heads[i]), int(relations[i]), int(tails[i])
            known = rt_to_heads.get((r, t), set())
            if known:
                mask_idx = torch.tensor(
                    [e for e in known if e != h], dtype=torch.long, device=device
                )
                if mask_idx.numel() > 0:
                    head_scores[i, mask_idx] = float("-inf")
            topk_scores, topk_idx = torch.topk(
                head_scores[i], min(k, num_entities)
            )
            head_records.append(
                {
                    "relation_id": r,
                    "tail_id": t,
                    "true_head_id": h,
                    "candidate_ids": topk_idx.cpu().tolist(),
                    "candidate_scores": topk_scores.cpu().tolist(),
                }
            )
 
    return tail_records, head_records
 
 
def compute_recall_at_k(records: List[dict], true_key: str, k_values: List[int]) -> Dict[int, float]:
    """Recall@k over an exported candidate list: fraction of queries where the
    true answer appears within the top-k candidates. This is the retriever's
    upper bound on what any downstream reranker (KICGPT, LiterallyLLM) can
    recover, and it is NOT reported by PyKEEN's standard evaluator or by any
    prior LitWD19K/48K paper — it has to be computed from records like these.
    """
    recall = {}
    n = len(records)
    for k in k_values:
        hits = sum(
            1
            for r in records
            if r[true_key] in r["candidate_ids"][:k]
        )
        recall[k] = hits / n if n else float("nan")
    return recall
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Path to LitWD19K or LitWD48K split files")
    parser.add_argument("--dataset-name", required=True, help="Label used in output filenames, e.g. litwd19k")
    parser.add_argument("--model", required=True, choices=["rotate", "tucker", "distmult"], help="PyKEEN model to train")
    parser.add_argument("--k", type=int, default=20, help="Candidate-set size exported for downstream reranking (default 20, matching FtG / mid-range of KICGPT's tuned k)")
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[1, 3, 10, 20], help="Additional k values to report Recall@k for")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for the candidate-export pass")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True, help="Directory to write metrics, candidates, and ID mappings to")
    parser.add_argument("--device", default=None, help="cuda / cpu; auto-detected if omitted")
    args = parser.parse_args()
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    # ---- 1. Load LitWD split as PyKEEN TriplesFactory objects ----
    train_tf, valid_tf, test_tf = load_litwd_triples_factories(args.dataset_dir)
    save_id_mappings(train_tf, out_dir)
    print(f"Loaded {args.dataset_name}: {train_tf.num_entities} entities, "
          f"{train_tf.num_relations} relations, "
          f"{train_tf.num_triples}/{valid_tf.num_triples}/{test_tf.num_triples} train/valid/test triples")
 
    # ---- 2. Train + standard filtered evaluation, via PyKEEN's pipeline() ----
    # This produces the aggregate MRR/Hits@1/3/10 rows for Table 2 for
    # whichever retriever you're currently running (RotatE row already
    # marked "+" in the paper; TuckER/DistMultLiteral rows likewise need
    # a real per-query candidate export, not just this aggregate number —
    # see the docstring at the top of this file).
    # Make PyKEEN's own evaluator report Hits@k at the same k values we care
    # about downstream (--k and --recall-ks), not just its default [1,3,5,10].
    # This is what lets you cross-check the custom Recall@k export below
    # against PyKEEN's own independently-computed number at the same k —
    # they should match, since both rank the full (filtered) entity set the
    # same way; if they don't, that's a bug in the manual export logic.
    #
    # NOTE: PyKEEN's RankBasedEvaluator does NOT take a top-level "ks"
    # kwarg (that was my first guess and it's wrong — confirmed by actually
    # running it: PyKEEN 1.11.1 raises TypeError: Evaluator.__init__() got
    # an unexpected keyword argument 'ks'). Custom k values are added as
    # extra HitsAtK metric instances instead, with add_defaults=True kept
    # so the standard MRR/Hits@1/3/5/10 metrics are still computed too.
    pykeen_ks = sorted(set([1, 3, 5, 10, args.k] + args.recall_ks))
    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model=args.model,
        model_kwargs=dict(embedding_dim=args.embedding_dim),
        training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        evaluator_kwargs=dict(
            metrics=["hits_at_k"] * len(pykeen_ks),
            metrics_kwargs=[{"k": k} for k in pykeen_ks],
            add_defaults=True,
        ),
        random_seed=args.random_seed,
        device=device,
    )
 
    metrics = result.metric_results.to_dict()
    with open(out_dir / f"{args.dataset_name}_{args.model}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Filtered evaluation metrics (PyKEEN's standard aggregate report):")
    print(json.dumps(metrics.get("both", metrics), indent=2)[:2000])
 
    result.save_to_directory(str(out_dir / f"{args.dataset_name}_{args.model}_pykeen_artifacts"))
 
    # ---- 3. Export filtered top-k candidates for every test triple ----
    # This is the artifact KICGPT / LiterallyLLM actually consume downstream.
    if any(rk > args.k for rk in args.recall_ks):
        raise ValueError(
            f"--recall-ks contains a value larger than --k ({args.k}): "
            f"{[rk for rk in args.recall_ks if rk > args.k]}. "
            "Recall@k for k beyond the exported candidate-set size is not "
            "meaningful — it would silently equal Recall@k(exported), not a "
            "true larger-k recall. Either raise --k to cover your largest "
            "--recall-ks value, or drop the offending values from --recall-ks."
        )
 
    model = result.model
    model = model.to(device)
    model.eval()
 
    hr_to_tails, rt_to_heads = build_known_triples_index(train_tf, valid_tf, test_tf)
    tail_records, head_records = export_filtered_candidates(
        model, test_tf, hr_to_tails, rt_to_heads, k=args.k, device=device, batch_size=args.eval_batch_size
    )
 
    with open(out_dir / f"{args.dataset_name}_{args.model}_candidates_tail.json", "w") as f:
        json.dump(tail_records, f)
    with open(out_dir / f"{args.dataset_name}_{args.model}_candidates_head.json", "w") as f:
        json.dump(head_records, f)
 
    # ---- 4. Recall@k — the retriever's upper bound for downstream reranking ----
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
    }
    with open(out_dir / f"{args.dataset_name}_{args.model}_recall_at_k.json", "w") as f:
        json.dump(recall_report, f, indent=2)
 
    print("\nRecall@k (retriever upper bound for downstream reranking):")
    print(json.dumps(recall_report, indent=2))
 
    print(f"\nAll outputs written to: {out_dir}")
 
 
if __name__ == "__main__":
    main()
 

