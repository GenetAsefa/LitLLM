"""
select_replicate.py

Picks ONE trained model out of the `--replicates` produced by
tune_retriever.py's final retrain step, to serve as the retriever fed into
export_from_tuned_model.py / the downstream LiterallyLLM pipeline -- as
opposed to the number(s) you report for RotatE as a baseline in a results
table, which should be the mean (and ideally std) of ALL replicates' test
metrics, not any single one of them.


USAGE
-----
    python select_replicate.py \\
        --replicates-dir out/litwd19k_rotate_hpo_best_replicates

    # One run prints mean/std/min/max of MRR, Hits@1, Hits@3 and Hits@10
    # (--report-keys, default) and selects the replicate by --metric-key
    # (default MRR). To select by another metric instead:
    python select_replicate.py \\
        --replicates-dir out/litwd19k_rotate_hpo_best_replicates \\
        --metric-key both.realistic.hits_at_10

"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_REPORT_KEYS = [
    "both.realistic.inverse_harmonic_mean_rank",
    "both.realistic.hits_at_1",
    "both.realistic.hits_at_3",
    "both.realistic.hits_at_10",
]


def get_metric(results: dict, dotted_key: str) -> float:
    """Walk a dotted key path (e.g. 'both.realistic.inverse_harmonic_mean_rank')
    into a nested results.json 'metrics' dict."""
    node = results["metrics"]
    for part in dotted_key.split("."):
        node = node[part]
    return float(node)


def find_replicates(replicates_dir: Path) -> list[Path]:
    root = replicates_dir / "replicates"
    if not root.is_dir():
        raise SystemExit(
            f"No 'replicates' subdirectory found under {replicates_dir}. "
            f"Pass the directory tune_retriever.py printed as "
            f"'<study_name>_best_replicates', not the 'replicates' "
            f"subdirectory itself, and not the top-level --out-dir."
        )
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("replicate-"))
    if not dirs:
        raise SystemExit(f"No replicate-XXXXX subdirectories found under {root}.")
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--replicates-dir",
        required=True,
        type=Path,
        help="The '<study_name>_best_replicates' directory tune_retriever.py printed at the end of the final retrain step (contains a 'replicates/' subdirectory).",
    )
    parser.add_argument(
        "--metric-key",
        default="both.realistic.inverse_harmonic_mean_rank",
        help="Dotted path into results.json's 'metrics' dict to rank replicates by. Default is test MRR, matching hpo_pipeline()'s own default objective (metric=None resolves to this exact key -- see module docstring). Use 'both.realistic.hits_at_10' for Hits@10 instead, etc.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Set this if --metric-key is a rank-based metric where LOWER is better (e.g. arithmetic_mean_rank). Not needed for MRR/Hits@k (higher is better), which is the default metric.",
    )
    parser.add_argument(
        "--selection",
        choices=["median", "best", "worst"],
        default="median",
        help="Which replicate to designate as the deployed retriever. 'median' (default, recommended) avoids test-set cherry-picking -- see module docstring for why 'best' is discouraged for this purpose.",
    )
    parser.add_argument(
        "--report-keys",
        nargs="+",
        default=DEFAULT_REPORT_KEYS,
        help="Dotted metric keys to summarize (mean/std/min/max across ALL replicates) in the baseline-table block. Default: MRR, Hits@1, Hits@3, Hits@10. This only affects the summary table; which replicate is selected is decided by --metric-key alone.",
    )
    args = parser.parse_args()

    replicate_dirs = find_replicates(args.replicates_dir)

    rows = []
    for d in replicate_dirs:
        results_path = d / "results.json"
        model_path = d / "trained_model.pkl"
        if not results_path.exists():
            print(f"WARNING: {results_path} missing, skipping {d.name}.")
            continue
        if not model_path.exists():
            print(f"WARNING: {model_path} missing (was --replicates run with save_replicates=True?), skipping {d.name}.")
            continue
        with open(results_path) as f:
            results = json.load(f)
        try:
            value = get_metric(results, args.metric_key)
        except KeyError as e:
            raise SystemExit(f"Metric key '{args.metric_key}' not found in {results_path}: missing {e}.")
        try:
            report = {k: get_metric(results, k) for k in args.report_keys}
        except KeyError as e:
            raise SystemExit(f"Report key not found in {results_path}: missing {e}.")
        rows.append(
            {
                "replicate": d.name,
                "results_path": str(results_path),
                "model_path": str(model_path),
                "value": value,
                "report_metrics": report,
            }
        )

    if not rows:
        raise SystemExit("No usable replicates found (see warnings above).")

    # Rank so index 0 = best.
    rows_sorted = sorted(rows, key=lambda r: r["value"], reverse=not args.lower_is_better)
    for rank, row in enumerate(rows_sorted, start=1):
        row["rank"] = rank

    values = [r["value"] for r in rows]
    n = len(rows_sorted)

    print(f"Metric: {args.metric_key} ({'lower' if args.lower_is_better else 'higher'} is better)")
    print(f"{n} replicate(s) found under {args.replicates_dir / 'replicates'}\n")
    print(f"{'rank':>4}  {'replicate':<16}  {'value':>10}")
    for row in rows_sorted:
        print(f"{row['rank']:>4}  {row['replicate']:<16}  {row['value']:>10.6f}")

    print(
        f"\nAcross all {n} replicate(s) -- USE THESE for your baseline-table row, "
        f"not any single replicate's number:"
    )
    print(f"  mean   = {statistics.mean(values):.6f}")
    if n > 1:
        print(f"  std    = {statistics.stdev(values):.6f}")
    print(f"  median = {statistics.median(values):.6f}")
    print(f"  min    = {min(values):.6f}")
    print(f"  max    = {max(values):.6f}")

    # Baseline-table block: mean +/- std (sample std, n-1) of every report
    # metric across ALL replicates, independent of which one gets selected.
    short = lambda k: k.split(".")[-1]
    print(f"\nBaseline-table summary over all {n} replicate(s) (std = sample std, n-1):")
    print(f"{'metric':<26} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}")
    report_summary = {}
    for k in args.report_keys:
        vals = [r["report_metrics"][k] for r in rows]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else None
        report_summary[k] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "values": vals}
        std_txt = f"{std:>9.4f}" if std is not None else f"{'n/a':>9}"
        print(f"{short(k):<26} {mean:>9.4f} {std_txt} {min(vals):>9.4f} {max(vals):>9.4f}")

    if args.selection == "best":
        chosen = rows_sorted[0]
    elif args.selection == "worst":
        chosen = rows_sorted[-1]
    else:  # median: statistics.median_low picks one real replicate, never an
        # average of two, and for an odd count (e.g. the default --replicates
        # 3) it's simply the unambiguous middle value.
        median_value = statistics.median_low(values)
        chosen = next(r for r in rows_sorted if r["value"] == median_value)

    print(
        f"\nSelected replicate ({args.selection}): {chosen['replicate']} "
        f"(rank {chosen['rank']} of {n}, {args.metric_key} = {chosen['value']:.6f})"
    )
    print(f"trained_model.pkl path (--trained-model-path for export_from_tuned_model.py):")
    print(f"  {chosen['model_path']}")

    selection_record = {
        "metric_key": args.metric_key,
        "lower_is_better": args.lower_is_better,
        "selection_method": args.selection,
        "replicates": rows_sorted,
        "aggregate": {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if n > 1 else None,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        },
        "report_summary": report_summary,
        "selected": chosen,
    }
    out_path = args.replicates_dir / "replicate_selection.json"
    with open(out_path, "w") as f:
        json.dump(selection_record, f, indent=2)
    print(f"\nFull selection record (keep this -- it's the audit trail for 'why this checkpoint'): {out_path}")


if __name__ == "__main__":
    main()