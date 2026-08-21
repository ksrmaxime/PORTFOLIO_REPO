# src/run6_plot.py
from __future__ import annotations

import pandas as pd

from src.run5_prompts import INSTRUMENT_CODES, TARGET_CODES
from src.run6_config import INSTRUMENT_ORDER, TARGET_DOMAINS, TARGET_ORDER


def build_portfolio_matrix(
    entries: pd.DataFrame,
    *,
    target_col: str = "target_code",
    instrument_col: str = "instrument_code",
    id_col: str = "row_id",
) -> pd.DataFrame:
    """
    Collapse the long-format Stage-2 output of run5 (one row per article x
    target x instrument entry) into the instrument x target count matrix
    that Figure 1 of the PoC visualizes: cell (i, t) = number of distinct
    articles for which instrument i was coded against target t.
    """
    for col in (target_col, instrument_col, id_col):
        if col not in entries.columns:
            raise KeyError(f"Missing column in entries file: {col}")

    unknown_targets = set(entries[target_col].dropna()) - set(TARGET_CODES)
    unknown_instruments = set(entries[instrument_col].dropna()) - set(INSTRUMENT_CODES)
    if unknown_targets:
        print(f"[run6] Warning: unknown target codes ignored: {sorted(unknown_targets)}")
    if unknown_instruments:
        print(f"[run6] Warning: unknown instrument codes ignored: {sorted(unknown_instruments)}")

    counts = (
        entries.groupby([instrument_col, target_col])[id_col]
        .nunique()
        .unstack(fill_value=0)
    )

    matrix = counts.reindex(index=INSTRUMENT_ORDER, columns=TARGET_ORDER, fill_value=0)
    matrix.index.name = "instrument_code"
    matrix.columns.name = "target_code"
    return matrix.astype(int)


def plot_portfolio_matrix(matrix: pd.DataFrame, *, title: str | None = None):
    """
    Render the instrument x target portfolio matrix as a grouped heatmap,
    reproducing the layout of Figure 1 in the PoC PDF: instruments on the Y
    axis, targets on the X axis grouped by their public-problem domain
    (Data / Skills / Infrastructure / Risk & Societal Harms), with each cell
    shaded and annotated by the number of coded articles it contains.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_rows, n_cols = matrix.shape
    data = matrix.to_numpy()

    fig_w = 2.6 + 0.65 * n_cols
    fig_h = 2.6 + 0.55 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = LinearSegmentedColormap.from_list("portfolio", ["#f7fbff", "#08306b"])
    vmax = max(int(data.max()), 1)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([TARGET_CODES[c] for c in matrix.columns], rotation=90, ha="center", fontsize=9)
    ax.set_yticklabels([INSTRUMENT_CODES[c] for c in matrix.index], fontsize=9)

    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if v <= 0:
                continue
            color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color=color)

    boundary = 0
    for codes in TARGET_DOMAINS.values():
        boundary += len(codes)
        if boundary < n_cols:
            ax.axvline(boundary - 0.5, color="#444444", linewidth=1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Coded articles", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        n_total = int(data.sum())
        ax.set_title(f"{title}\n(n = {n_total:,} instrument-target entries)", fontsize=11, pad=12)

    fig.subplots_adjust(bottom=0.4, right=0.98, top=0.9)

    # Domain group separators + labels, positioned below the actual rendered
    # extent of the (rotated, variable-length) column tick labels — measured
    # via the renderer rather than a fixed offset, since label length varies
    # with the target/instrument taxonomy.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    min_y_axes = 1.0
    for lbl in ax.get_xticklabels():
        bbox = lbl.get_window_extent(renderer=renderer)
        y0_axes = inv.transform((0, bbox.y0))[1]
        min_y_axes = min(min_y_axes, y0_axes)

    trans = ax.get_xaxis_transform()
    line_y = min_y_axes - 0.04
    label_y = line_y - 0.02

    boundary = 0
    for domain, codes in TARGET_DOMAINS.items():
        start, end = boundary, boundary + len(codes)
        ax.plot(
            [start - 0.5, end - 0.5], [line_y, line_y],
            color="#333333", linewidth=1.0, transform=trans, clip_on=False,
        )
        ax.text(
            (start + end - 1) / 2, label_y, domain,
            transform=trans, ha="center", va="top", fontsize=9.5, fontweight="bold", clip_on=False,
        )
        boundary = end

    return fig
