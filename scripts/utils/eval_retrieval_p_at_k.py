"""
Task 3 — Retrieval quality evaluation for labeled 15×20 sheets.

Modes:
1) Single-file:
   python scripts/eval_retrieval_p_at_k.py --single "data/labels_filled_15x20.csv" --name old_relabel

2) Comparison (baseline vs v2):
   python scripts/eval_retrieval_p_at_k.py --baseline "data/labels_filled_15x20.csv" --v2 "data/labels_filled_15x20_v2.csv"

Outputs go to: results/task3 (by default)

Fix included:
- Removes ambiguity where question_id existed both as index and column (pandas merge error).
"""

import argparse
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _to_rel(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return 0
    if s in {"true", "yes", "y"}:
        return 1
    if s in {"false", "no", "n"}:
        return 0
    try:
        v = float(s)
        return 1 if v >= 1.0 else 0
    except Exception:
        return 0


def load_labels_csv(path: Path, label_col: str = "final_label") -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"question_id", "chunk_id", label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # Define rank by file order within each question
    df["_row"] = range(len(df))
    df = df.sort_values(["question_id", "_row"]).copy()
    df["rank"] = df.groupby("question_id").cumcount() + 1

    df["rel"] = df[label_col].apply(_to_rel).astype(int).clip(0, 1)
    return df


def metrics_at_k(df: pd.DataFrame, k: int, pool_k: int = 20) -> pd.DataFrame:
    pool = df[df["rank"] <= pool_k].copy()
    topk = pool[pool["rank"] <= k].copy()

    rel_topk = topk.groupby("question_id")["rel"].sum()
    rel_pool = pool.groupby("question_id")["rel"].sum()

    # Build as columns, then FORCE a clean RangeIndex to avoid merge ambiguity
    out = pd.DataFrame({
        "question_id": rel_pool.index.astype(str),
        f"P@{k}": (rel_topk / k).reindex(rel_pool.index).fillna(0).to_numpy(),
        f"R@{k} (within top{pool_k})": (rel_topk / rel_pool.replace(0, pd.NA)).reindex(rel_pool.index).fillna(0).to_numpy(),
        f"rel_in_top{k}": rel_topk.reindex(rel_pool.index).fillna(0).astype(int).to_numpy(),
        f"rel_in_top{pool_k}": rel_pool.reindex(rel_pool.index).fillna(0).astype(int).to_numpy(),
    }).reset_index(drop=True)

    return out


def write_single_outputs(metrics_df: pd.DataFrame, outdir: Path, name: str, k: int, pool_k: int):
    metrics_path = outdir / f"retrieval_metrics_{name}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    summary_path = outdir / f"retrieval_summary_{name}.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Retrieval quality — {name}\n\n")
        f.write(f"- Questions: {metrics_df['question_id'].nunique()}\n")
        f.write(f"- Mean P@{k}: {metrics_df[f'P@{k}'].mean():.3f}\n")
        f.write(f"- Mean R@{k} (within top{pool_k}): {metrics_df[f'R@{k} (within top{pool_k})'].mean():.3f}\n")

    print("Saved:")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--baseline", default=str(BASE_DIR / "data" / "labels" / "labels_filled_15x20.csv"))
    ap.add_argument("--v2", default=str(BASE_DIR / "data" / "labels" / "labels_filled_15x20_v2.csv"))
    ap.add_argument("--outdir", default=str(BASE_DIR / "results" / "task3"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool_k", type=int, default=20)

    ap.add_argument("--single", default=None, help="Evaluate only ONE labels CSV (no comparison).")
    ap.add_argument("--name", default="run", help="Output name for --single mode (e.g., old_relabel, v2).")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.single is not None:
        in_path = Path(args.single)
        if not in_path.exists():
            raise FileNotFoundError(f"Labels file not found: {in_path}")
        df = load_labels_csv(in_path)
        m = metrics_at_k(df, k=args.k, pool_k=args.pool_k)
        write_single_outputs(m, outdir, args.name, args.k, args.pool_k)
        return

    baseline_path = Path(args.baseline)
    v2_path = Path(args.v2)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline labels file not found: {baseline_path}")
    if not v2_path.exists():
        raise FileNotFoundError(f"V2 labels file not found: {v2_path}")

    b = load_labels_csv(baseline_path)
    v = load_labels_csv(v2_path)

    b_m = metrics_at_k(b, k=args.k, pool_k=args.pool_k)
    v_m = metrics_at_k(v, k=args.k, pool_k=args.pool_k)

    comp = b_m.merge(v_m, on="question_id", suffixes=("_baseline", "_v2"))
    comp[f"delta_P@{args.k}"] = comp[f"P@{args.k}_v2"] - comp[f"P@{args.k}_baseline"]
    comp[f"delta_R@{args.k}"] = comp[f"R@{args.k} (within top{args.pool_k})_v2"] - comp[f"R@{args.k} (within top{args.pool_k})_baseline"]

    comp_path = outdir / "retrieval_metrics_p5_r5_comparison.csv"
    comp.to_csv(comp_path, index=False, encoding="utf-8")

    summary_path = outdir / "task3_retrieval_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Task 3 — Retrieval quality (baseline vs v2)\n\n")
        f.write(f"- Questions: {comp['question_id'].nunique()}\n")
        f.write(f"- Mean P@{args.k} baseline: {comp[f'P@{args.k}_baseline'].mean():.3f}\n")
        f.write(f"- Mean P@{args.k} v2: {comp[f'P@{args.k}_v2'].mean():.3f}\n")
        f.write(f"- Mean ΔP@{args.k}: {comp[f'delta_P@{args.k}'].mean():.3f}\n")
        f.write(f"- Mean R@{args.k} (within top{args.pool_k}) baseline: {comp[f'R@{args.k} (within top{args.pool_k})_baseline'].mean():.3f}\n")
        f.write(f"- Mean R@{args.k} (within top{args.pool_k}) v2: {comp[f'R@{args.k} (within top{args.pool_k})_v2'].mean():.3f}\n")
        f.write(f"- Mean ΔR@{args.k}: {comp[f'delta_R@{args.k}'].mean():.3f}\n")

    print("Saved:")
    print(f"- {comp_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
