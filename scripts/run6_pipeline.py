import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.run6_plot import build_portfolio_matrix, plot_portfolio_matrix


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render the final AI regulation portfolio matrix (Figure 1 of the PoC) "
        "from the long-format portfolio entries produced by run5."
    )

    ap.add_argument(
        "--input",
        required=True,
        help="Path to the run5 output, e.g. "
        "<output_base>_entries_job<RUN5_JOB_ID>.parquet (or .csv).",
    )
    ap.add_argument("--output_base", required=True)

    ap.add_argument("--target_col", default="target_code")
    ap.add_argument("--instrument_col", default="instrument_code")
    ap.add_argument("--id_col", default="row_id")

    ap.add_argument("--title", default="AI Regulation Portfolio — Switzerland")

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    print(f"Loaded {len(df):,} portfolio entries from {args.input}")

    if df.empty:
        print("[run6] Warning: input has no rows — the portfolio matrix will be empty.")

    matrix = build_portfolio_matrix(
        df,
        target_col=args.target_col,
        instrument_col=args.instrument_col,
        id_col=args.id_col,
    )

    out_base = Path(args.output_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    matrix_csv = f"{out_base}_matrix.csv"
    matrix.to_csv(matrix_csv)

    fig = plot_portfolio_matrix(matrix, title=args.title)
    pdf_path = f"{out_base}.pdf"
    png_path = f"{out_base}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    n_total = int(matrix.to_numpy().sum())
    n_filled = int((matrix.to_numpy() > 0).sum())
    print(f"Portfolio entries plotted: {n_total:,} across {n_filled}/{matrix.size} instrument x target cells")
    print(f"Saved: {pdf_path}, {png_path}, {matrix_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
