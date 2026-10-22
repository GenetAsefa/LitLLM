"""
tune_retriever.py

Hyperparameter-tunes a base KGE retriever (RotatE by default) on LitWD1K /
LitWD19K / LitWD48K using PyKEEN's built-in HPO module (Optuna under the
hood), then retrains the best configuration multiple times ("replicates")
and reports test-set metrics for each replicate.


Two search modes:
  1. Default (Bayesian/TPE, Optuna's normal sampler): draws from the SAME
     value sets as the paper (categorical, not a smooth range), but doesn't
     exhaustively try every combination -- it's a directed search, cheaper
     than the paper's actual method, at the cost of not being a literal
     reproduction of "grid search."
  2. --grid-search: an actual grid search over the Cartesian product of
     embedding_dim x batch_size x margin x adversarial_temperature (4 x 3 x
     7 x 2 = 168 combinations by default), using Optuna's GridSampler. This
     is what the paper's own method actually was. --negs-per-pos is already
     fixed by default (256), so it needs no special handling here.
     n_trials is computed automatically from the grid size and overrides
     --n-trials.


Usage (default, Bayesian search over the paper's value sets; negs-per-pos
fixed at 256, lr searched log-uniform in [1e-5, 5e-4]; final retrain at
800 epochs, 2x the 400-epoch trial ceiling):
    python tune_retriever.py \
        --dataset-dir /path/to/LiterallyWikidata/Datasets/LitWD19K \
        --dataset-name litwd19k \
        --model rotate \
        --n-trials 50 \
        --max-epochs 400 \
        --replicates 3 \
        --out-dir ./hpo_outputs/litwd19k_rotate \
        --device cuda

Usage (faithful grid search over the paper's exact 168-combination grid;
LR pinned to FB15k-237's own value -- --negs-per-pos already defaults to
256; --final-epochs pinned explicitly here to match LiterallyWikidata's
literal 500->1000 ratio rather than the generic 2x default):
    python tune_retriever.py \
        --dataset-dir /path/to/LiterallyWikidata/Datasets/LitWD19K \
        --dataset-name litwd19k \
        --model rotate \
        --grid-search --lr 5e-5 \
        --max-epochs 400 \
        --final-epochs 800 \
        --replicates 3 \
        --out-dir ./hpo_outputs/litwd19k_rotate_grid \
        --device cuda

Usage (grid search with LR actually searched, not pinned -- 840
combinations: the 168 above x 5 LR values spanning the dataset-native
[0.001, 0.1] range from Bringing-Light-Into-the-Dark/LiterallyWikidata
instead of trusting FB15k-237's borrowed value):
    python tune_retriever.py \
        --dataset-dir /path/to/LiterallyWikidata/Datasets/LitWD19K \
        --dataset-name litwd19k \
        --model rotate \
        --grid-search --lr-grid-values 0.001 0.003 0.01 0.03 0.1 \
        --max-epochs 400 \
        --final-epochs 800 \
        --replicates 3 \
        --out-dir ./hpo_outputs/litwd19k_rotate_grid_lrsearch \
        --device cuda

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pykeen.hpo import hpo_pipeline
from pykeen.hpo.hpo import replicate_pipeline_from_config


def export_trials_csv(study, path, max_epochs: int, stopper_frequency: int, stopper_patience: int) -> None:
    """Write one row per Optuna trial to a CSV (the .db is SQLite, not readable
    directly). Everything here is already stored by Optuna/PyKEEN per trial;
    this only flattens it.

    """
    import csv

    def metric(attrs, name):
        return attrs.get(f"both.realistic.{name}", attrs.get(name))

    trials = list(study.trials)
    param_names = sorted({p for t in trials for p in t.params})
    header = (
        ["trial", "state"]
        + [f"param.{p}" for p in param_names]
        + ["val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10",
           "stopped_epoch", "approx_best_epoch", "end_reason", "duration_s", "random_seed"]
    )
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in trials:
            a = t.user_attrs
            stopped = a.get("stopped_epoch")
            if t.state.name != "COMPLETE":
                reason = t.state.name.lower()
            elif stopped is not None:
                reason = "early_stopping"
            else:
                reason = f"reached_max_epochs({max_epochs})"
            approx_best = (
                max(0, stopped - stopper_patience * stopper_frequency) if stopped is not None else ""
            )
            w.writerow(
                [t.number, t.state.name]
                + [t.params.get(p, "") for p in param_names]
                + [t.value if t.value is not None else "",
                   metric(a, "hits_at_1") if metric(a, "hits_at_1") is not None else "",
                   metric(a, "hits_at_3") if metric(a, "hits_at_3") is not None else "",
                   metric(a, "hits_at_10") if metric(a, "hits_at_10") is not None else "",
                   stopped if stopped is not None else "",
                   approx_best,
                   reason,
                   f"{t.duration.total_seconds():.1f}" if t.duration is not None else "",
                   a.get("random_seed", "")]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Path to LitWD1K/19K/48K split files")
    parser.add_argument("--dataset-name", required=True, help="Label used in output filenames/study name")
    parser.add_argument("--model", default="rotate", choices=["rotate", "tucker", "distmult"])
    parser.add_argument("--train-filename", default="train.txt")
    parser.add_argument("--valid-filename", default="valid.txt")
    parser.add_argument("--test-filename", default="test.txt")

    # Paper-exact search values (Sun et al., 2019). Only meaningfully used
    # for --model rotate; for tucker/distmult these are ignored and PyKEEN's
    # own defaults apply instead .
    parser.add_argument("--embedding-dims", type=int, nargs="+", default=[125, 250, 500, 1000])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--margins", type=int, nargs="+", default=[3, 6, 9, 12, 18, 24, 30])
    parser.add_argument("--adversarial-temperatures", type=float, nargs="+", default=[0.5, 1.0])

    # Not in the RotatE paper's text -- taken from the authors' own released
    # code (best_config.sh / run.sh) instead. See module docstring for the
    # full reasoning on why negs-per-pos is FIXED and lr is only NARROWED.
    parser.add_argument("--lr-low", type=float, default=1e-5, help="Ignored if --lr or --lr-grid-values is set. Brackets FB15k-237's actual value (5e-5). NOT usable with --grid-search (continuous range, GridSampler needs a finite list) -- use --lr-grid-values there instead.")
    parser.add_argument("--lr-high", type=float, default=5e-4, help="Ignored if --lr or --lr-grid-values is set. Brackets FB15k-237's actual value (5e-5). NOT usable with --grid-search -- see --lr-low.")
    parser.add_argument("--lr", type=float, default=None, help="Fix learning rate to a single value instead of searching it (matches the original RotatE paper's own approach -- it never searched LR either). One of two ways to satisfy --grid-search's LR requirement; see --lr-grid-values for the other (LR actually searched, as a discrete grid dimension). Mutually exclusive with --lr-grid-values.")
    parser.add_argument("--lr-grid-values", type=float, nargs="+", default=None, help="Search learning rate as a genuine grid dimension over this explicit list of candidate values (e.g. --lr-grid-values 0.001 0.003 0.01 0.03 0.1), instead of pinning a single value with --lr. Only meaningful with --grid-search -- GridSampler needs the list declared upfront, unlike the default TPE mode's continuous --lr-low/--lr-high range. Multiplies the grid size by len(this list). Mutually exclusive with --lr. See module docstring for why [0.001, 0.1] log-spaced is the evidence-grounded range to draw values from (Bringing-Light-Into-the-Dark Table 6 + LiterallyWikidata Table 4, independently identical).")
    parser.add_argument("--negs-per-pos", type=int, default=256, help="Negative samples per positive triple, FIXED (not searched) -- default 256 is FB15k-237's value from the original authors' best_config.sh, held constant across LitWD1K/19K/48K so it can't confound a cross-scale comparison. Disclosed simplification, not a claim of optimality: for RotatE's distance-based scoring function, reaching the theoretical optimum under uniform sampling needs n >= |entities| (arXiv:2206.10140), which no feasible fixed value or search range reaches anyway -- the principled fix is the LCWA arm (exhaustive scoring over all entities), not a wider search here. See module docstring for full reasoning, including comparison to Bringing-Light-Into-the-Dark's own documented {1..100} search range for this parameter. Override to fix a different value.")

    parser.add_argument("--grid-search", action="store_true", help="Use Optuna's GridSampler for an exhaustive search over --embedding-dims x --batch-sizes x --margins x --adversarial-temperatures (plus --lr-grid-values if given), matching the paper's actual method (rather than the default Bayesian/TPE search over the same value sets). Requires LR to be finite/enumerable: pass either --lr (fixed, not searched) or --lr-grid-values (searched as a discrete list) -- the default continuous --lr-low/--lr-high range doesn't work here. --negs-per-pos is already fixed by default (256), so it needs no extra handling.")

    parser.add_argument("--max-epochs", type=int, default=400, help="Upper bound on epochs per trial; early stopping usually cuts this short. Default approximates FB15k-237's original 100k-step budget at batch size 1024 (~376 epochs) -- see module docstring for why this is approximate, not exact, for LitWD.")
    parser.add_argument("--final-epochs", type=int, default=None, help="Epoch budget for the FINAL retrain of the best config (the --replicates step), decoupled from --max-epochs. Defaults to 2x --max-epochs, mirroring LiterallyWikidata's own 500->1000 trial-vs-retrain split for LitWD19K/48K. Without this, PyKEEN's replicate_best_pipeline() would silently retrain for however many epochs early stopping cut the WINNING HPO TRIAL to -- not --max-epochs, and not a bigger budget -- which matches neither reference paper's methodology. See code comment above the retrain call for details.")
    parser.add_argument("--stopper-frequency", type=int, default=10, help="Evaluate on validation every N epochs for early stopping")
    parser.add_argument("--stopper-patience", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials for the default (non-grid) search. Ignored and auto-computed when --grid-search is set.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional wall-clock budget for the whole search, in seconds")
    parser.add_argument("--replicates", type=int, default=3, help="How many times to retrain the BEST config with different random seeds, for a mean +/- std test-set result rather than a single lucky/unlucky run")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-seed", type=int, default=None, help="Optuna sampler seed, for reproducible trial ordering (does NOT fix each trial's own model seed)")
    parser.add_argument("--fresh-study", action="store_true", help="Delete any existing Optuna study .db for this --dataset-name/--model/--out-dir before running, instead of resuming it. Use this whenever you've changed the search space (any of --embedding-dims/--batch-sizes/--margins/--adversarial-temperatures/--lr/--lr-low/--lr-high/--lr-grid-values/--negs-per-pos/--grid-search) since the last run with this same --out-dir -- resuming across a changed search space is what causes Optuna's 'Cannot set different distribution kind to the same parameter name' error.")
    args = parser.parse_args()

    if args.lr is not None and args.lr_grid_values is not None:
        parser.error(
            "--lr and --lr-grid-values are mutually exclusive: pick a single fixed "
            "learning rate (--lr) or a discrete searched set (--lr-grid-values), not both."
        )

    if args.grid_search and args.lr is None and args.lr_grid_values is None:
        parser.error(
            "--grid-search needs a finite, explicitly enumerable value for LR, because "
            "Optuna's GridSampler requires every tuned parameter's search space to be "
            "declared upfront as a list -- it can't handle the default continuous "
            "--lr-low/--lr-high log-uniform range. Two ways to satisfy this:\n"
            "  (1) --lr VALUE -- pin a single fixed learning rate, not searched at all "
            "(matches the original RotatE paper's own approach; e.g. --lr 5e-5, "
            "FB15k-237's value).\n"
            "  (2) --lr-grid-values V1 V2 ... -- search LR as a genuine grid dimension "
            "over a short discrete list (e.g. --lr-grid-values 0.001 0.003 0.01 0.03 0.1, "
            "matching the log-scale [0.001, 0.1] range both Bringing-Light-Into-the-Dark "
            "and the LiterallyWikidata paper independently searched -- see module "
            "docstring). This multiplies the grid size by len(--lr-grid-values)."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    train_path = str(dataset_dir / args.train_filename)
    valid_path = str(dataset_dir / args.valid_filename)
    test_path = str(dataset_dir / args.test_filename)

    model_kwargs_ranges = dict(
        embedding_dim=dict(type="categorical", choices=args.embedding_dims)
    )
    training_kwargs = {}
    training_kwargs_ranges = dict(batch_size=dict(type="categorical", choices=args.batch_sizes))
    # --max-epochs is a fixed per-trial CEILING in BOTH search modes; early
    # stopping does the real work of cutting a trial short if it plateaus.
    # (An earlier version searched num_epochs as an integer in
    # [max_epochs/10, max_epochs] in TPE mode. That was a bug: every trial then
    # drew its own random ceiling -- e.g. 195 -- so trials were truncated
    # arbitrarily and compared unfairly, and TPE was nudged toward "more
    # epochs = better MRR" instead of toward better hyperparameters. GridSampler
    # also requires a fixed value here, so both modes now share this.)
    training_kwargs["num_epochs"] = args.max_epochs

    optimizer_kwargs = {}
    optimizer_kwargs_ranges = {}
    if args.lr_grid_values is not None:
        optimizer_kwargs_ranges["lr"] = dict(type="categorical", choices=args.lr_grid_values)
    elif args.lr is not None:
        optimizer_kwargs["lr"] = args.lr
    else:
        optimizer_kwargs_ranges["lr"] = dict(type=float, low=args.lr_low, high=args.lr_high, log=True)

    negative_sampler_kwargs = {}
    negative_sampler_kwargs_ranges = {}
    if args.negs_per_pos is not None:
        negative_sampler_kwargs["num_negs_per_pos"] = args.negs_per_pos
    # else: leave both empty -> PyKEEN auto-applies BasicNegativeSampler.hpo_default
    # (log-uniform 1-100). Not expected to trigger given --negs-per-pos now
    # defaults to 256; kept only in case a future caller explicitly passes
    # --negs-per-pos as unset via code rather than the CLI.

    # RotatE-specific: override PyKEEN's plain default loss (MarginRankingLoss)
    # with the self-adversarial loss the original paper actually uses, and
    # pin its search values to the paper's exact grid rather than PyKEEN's
    # own (differently-spaced) NSSALoss.hpo_default.
    loss_extra = {}
    if args.model == "rotate":
        loss_extra["loss"] = "NSSALoss"
        loss_extra["loss_kwargs_ranges"] = dict(
            margin=dict(type="categorical", choices=args.margins),
            adversarial_temperature=dict(type="categorical", choices=args.adversarial_temperatures),
        )

    study_name = f"{args.dataset_name}_{args.model}_hpo" + ("_grid" if args.grid_search else "")
    db_path = out_dir / (study_name + ".db")
    storage = f"sqlite:///{db_path}"

    if args.fresh_study and db_path.exists():
        # Also remove SQLite's -wal/-shm sidecar files if present, so no
        # stale write-ahead-log fragments survive to confuse the new study.
        for p in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
            if p.exists():
                p.unlink()
        print(f"--fresh-study: deleted existing study database at {db_path}")

    sampler = None
    sampler_kwargs = None
    n_trials = args.n_trials
    if args.grid_search:
        sampler = "grid"
        search_space = {
            "model.embedding_dim": args.embedding_dims,
            "training.batch_size": args.batch_sizes,
        }
        if args.model == "rotate":
            search_space["loss.margin"] = args.margins
            search_space["loss.adversarial_temperature"] = args.adversarial_temperatures
        if args.lr_grid_values is not None:
            search_space["optimizer.lr"] = args.lr_grid_values
        sampler_kwargs = dict(search_space=search_space)
        n_trials = 1
        for values in search_space.values():
            n_trials *= len(values)
        print(f"--grid-search: exhaustive grid of {n_trials} combinations "
              f"({' x '.join(f'{len(v)} {k}' for k, v in search_space.items())})")
    elif args.random_seed is not None:
        sampler_kwargs = dict(seed=args.random_seed)

    try:
        result = hpo_pipeline(
            training=train_path,
            validation=valid_path,
            testing=test_path,
            model=args.model,
            model_kwargs_ranges=model_kwargs_ranges,
            optimizer="Adam",
            optimizer_kwargs=optimizer_kwargs or None,
            optimizer_kwargs_ranges=optimizer_kwargs_ranges or None,
            training_kwargs=training_kwargs or None,
            training_kwargs_ranges=training_kwargs_ranges,
            negative_sampler_kwargs=negative_sampler_kwargs or None,
            negative_sampler_kwargs_ranges=negative_sampler_kwargs_ranges or None,
            stopper="early",
            stopper_kwargs=dict(
                # Early stopping must watch the SAME metric the search optimizes
                # (validation MRR). PyKEEN's EarlyStopper defaults to Hits@10
                # ("hits_at_k"), which would stop trials and restore checkpoints
                # on Hits@10 while each trial is scored on MRR. LiterallyWikidata
                # also stopped early on validation MRR (Sec. 5.2).
                metric="both.realistic.inverse_harmonic_mean_rank",
                frequency=args.stopper_frequency,
                patience=args.stopper_patience,
                relative_delta=0.01,
            ),
            # PyKEEN's default pruner is Optuna's MedianPruner (5 startup trials,
            # no warm-up): after 5 trials it kills any trial whose validation MRR
            # at an evaluation is below the median of earlier trials at that step,
            # even at the first evaluation. That (a) silently turns an "exhaustive"
            # grid into one where pruned grid points are never fully trained or
            # scored, and (b) biases the search against slow-starting settings
            # such as lr=1e-4. Neither reference paper describes pruning, so it is
            # disabled: every trial runs until early stopping or --max-epochs.
            pruner="nop",
            n_trials=n_trials,
            timeout=args.timeout,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            device=args.device,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            **loss_extra,
        )
    except ValueError as e:
        if "different distribution kind" in str(e):
            raise SystemExit(
                f"\nOptuna refused to resume the existing study '{study_name}' "
                f"({db_path}): {e}\n\n"
                "This means the search space (any of --embedding-dims / "
                "--batch-sizes / --margins / --adversarial-temperatures / --lr / "
                "--lr-low / --lr-high / --lr-grid-values / --negs-per-pos / "
                "--grid-search) differs "
                "from whatever produced the trials already stored in that .db "
                "file -- Optuna requires every trial in a study to use the exact "
                "same distribution for a given parameter name, so a search-space "
                "change can't be resumed into, only started fresh.\n"
                "Fix: rerun with --fresh-study to delete and restart this study "
                "(the old trials will be lost), or point --out-dir/--dataset-name "
                "somewhere new to keep both."
            ) from e
        raise

    print("Best trial parameters (validation MRR):")
    print(json.dumps(result.study.best_params, indent=2))
    print(f"Best validation MRR: {result.study.best_value}")

    with open(out_dir / f"{study_name}_best_params.json", "w") as f:
        json.dump({"best_params": result.study.best_params, "best_value_mrr": result.study.best_value}, f, indent=2)

    trials_csv = out_dir / f"{study_name}_trials.csv"
    export_trials_csv(result.study, trials_csv, args.max_epochs, args.stopper_frequency, args.stopper_patience)
    print(f"Per-trial table ({len(result.study.trials)} trials): {trials_csv}")

  
    final_epochs = args.final_epochs if args.final_epochs is not None else 2 * args.max_epochs
    best_config = result._get_best_study_config()
    stopped_at = best_config["pipeline"]["training_kwargs"].get("num_epochs")
    best_config["pipeline"]["training_kwargs"]["num_epochs"] = final_epochs
    print(
        f"\nFinal retrain epochs: {final_epochs} "
        f"(overriding {stopped_at}, the epoch PyKEEN's early stopping cut the "
        f"winning HPO trial to -- NOT --max-epochs={args.max_epochs}, and NOT "
        f"reused as-is; see module docstring / code comment above for why)."
    )
    with open(out_dir / f"{study_name}_final_retrain_config.json", "w") as f:
        json.dump(best_config, f, indent=2, default=str)

    replicate_dir = out_dir / f"{study_name}_best_replicates"
    replicate_pipeline_from_config(
        config=best_config,
        directory=str(replicate_dir),
        replicates=args.replicates,
        use_testing_data=True,
        save_replicates=True,
    )
    print(f"\n{args.replicates} replicate(s) of the best configuration retrained and evaluated on the test split.")
    print(f"Per-replicate trained models + results.json: {replicate_dir / 'replicates'}")
    print(
        "Next step: feed one of these trained_model.pkl files into "
        "export_from_tuned_model.py to produce the actual top-k candidate "
        "export + Recall@k that KICGPT / LiterallyLLM consume downstream -- "
        "the HPO search itself only reports aggregate metrics, it doesn't "
        "export candidate lists."
    )


if __name__ == "__main__":
    main()