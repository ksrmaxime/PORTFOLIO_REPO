# src/run6_plot.py
from __future__ import annotations

import re

import pandas as pd

from src.run5_prompts import INSTRUMENT_CODES
from src.run6_config import (
    INSTRUMENT_ORDER,
    TARGET_CODES,
    TARGET_DISPLAY_NAMES,
    TARGET_FUNCTION_RUNS,
    TARGET_LOCATION_RUNS,
    TARGET_ORDER,
    TARGET_QUADRANT_RUNS,
)

# Values treated as a positive ("OUI"/True) decision when reading a wide
# gold/prediction table (PORTFOLIO_GOLD.csv, or a merged run_target/run_inst
# output). Covers every encoding used across the pipeline: Python booleans
# (str "True"/"False"), the model's French OUI/NON, and common 0/1 / Yes-No
# spreadsheet conventions.
_TRUTHY = {"TRUE", "T", "OUI", "YES", "Y", "1"}


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().upper() in _TRUTHY


def build_portfolio_matrix(
    entries: pd.DataFrame,
    *,
    target_col: str = "target_code",
    instrument_col: str = "instrument_code",
    id_col: str = "row_id",
) -> pd.DataFrame:
    """
    Collapse a long-format table (one row per article x target x instrument
    entry — the legacy run5 "entries" output) into the instrument x target
    count matrix that Figure 1 of the PoC visualizes: cell (i, t) = number of
    distinct articles for which instrument i was coded against target t.
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


def _find_wide_columns(columns: list[str], codes: list[str], prefixes: list[str]) -> dict[str, str]:
    """Map each code to the actual column name in `columns` that encodes it,
    trying each of `prefixes` case-insensitively (e.g. "Target_" then
    "target_" for code "RESEARCH_INNOVATION")."""
    lookup = {c.lower(): c for c in columns}
    found: dict[str, str] = {}
    for code in codes:
        for prefix in prefixes:
            candidate = f"{prefix}{code}".lower()
            if candidate in lookup:
                found[code] = lookup[candidate]
                break
    return found


def build_portfolio_matrix_from_wide(
    df: pd.DataFrame,
    *,
    target_prefixes: tuple[str, ...] = ("Target_", "target_"),
    instrument_prefixes: tuple[str, ...] = ("Instrument_", "instrument_"),
) -> pd.DataFrame:
    """
    Build the instrument x target count matrix directly from a wide table
    with one row per article and one boolean-like column per target
    (Target_<CODE>) and per instrument (Instrument_<CODE>) — the format of
    PORTFOLIO_GOLD.csv and of the human-labelled benchmark used by
    sbatch_run_target.sh / sbatch_run_inst.sh for scoring.

    Cell (i, t) = number of articles for which BOTH target t and instrument i
    are coded True, i.e. every confirmed target is crossed with every
    confirmed instrument found in the same article (PoC PDF, section 4.3.3).
    """
    columns = list(df.columns)
    target_cols = _find_wide_columns(columns, TARGET_ORDER, list(target_prefixes))
    instrument_cols = _find_wide_columns(columns, INSTRUMENT_ORDER, list(instrument_prefixes))

    missing_targets = [c for c in TARGET_ORDER if c not in target_cols]
    missing_instruments = [c for c in INSTRUMENT_ORDER if c not in instrument_cols]
    if missing_targets:
        raise KeyError(
            f"Missing target columns in wide input (tried prefixes {target_prefixes}): {missing_targets}"
        )
    if missing_instruments:
        raise KeyError(
            f"Missing instrument columns in wide input (tried prefixes {instrument_prefixes}): {missing_instruments}"
        )

    target_bool = pd.DataFrame({code: df[col].map(_is_truthy) for code, col in target_cols.items()})
    instrument_bool = pd.DataFrame({code: df[col].map(_is_truthy) for code, col in instrument_cols.items()})

    # (instrument x target) cross count = instrument_bool^T @ target_bool,
    # i.e. for each article, every True instrument is crossed with every
    # True target it also carries.
    matrix = instrument_bool.astype(int).T.dot(target_bool.astype(int))
    matrix = matrix.reindex(index=INSTRUMENT_ORDER, columns=TARGET_ORDER, fill_value=0)
    matrix.index.name = "instrument_code"
    matrix.columns.name = "target_code"
    return matrix.astype(int)


def detect_and_build_portfolio_matrix(
    df: pd.DataFrame,
    *,
    target_col: str = "target_code",
    instrument_col: str = "instrument_code",
    id_col: str = "row_id",
) -> pd.DataFrame:
    """Dispatch to the long- or wide-format builder depending on which
    columns are actually present in `df`."""
    if target_col in df.columns and instrument_col in df.columns:
        return build_portfolio_matrix(df, target_col=target_col, instrument_col=instrument_col, id_col=id_col)
    return build_portfolio_matrix_from_wide(df)


# ---------------------------------------------------------------------------
# Sequential blue ramp (light -> dark), validated for heatmap/magnitude use.
# ---------------------------------------------------------------------------
_SEQUENTIAL_BLUE = ["#f7fafd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_GRID_LINE = "#dedcd6"
_QUADRANT_LINE = "#8a8878"
_BAND_LINE = "#0b0b0b"


def plot_portfolio_matrix(matrix: pd.DataFrame, *, title: str | None = None):
    """
    Render the instrument x target portfolio matrix as a heatmap laid out
    like Figure 1 of the PoC PDF: 7 instruments on the Y axis; 10 targets on
    the X axis, grouped by a nested Enabling/Safeguarding x Upstream/
    Downstream header that mirrors the paper's conceptual grid instead of a
    single flat row of ad hoc domain labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_rows, n_cols = matrix.shape
    data = matrix.to_numpy()

    fig_w = 2.8 + 0.75 * n_cols
    fig_h = 3.6 + 0.5 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = LinearSegmentedColormap.from_list("portfolio_blue", _SEQUENTIAL_BLUE)
    vmax = max(int(data.max()), 1)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(
        [TARGET_DISPLAY_NAMES[c] for c in matrix.columns],
        rotation=90, ha="center", va="top", fontsize=9, color=_INK,
    )
    ax.set_yticklabels([INSTRUMENT_CODES[c] for c in matrix.index], fontsize=9.5, color=_INK)

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
            color = "white" if v > vmax * 0.6 else _INK
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5, color=color)

    # Heavier separators at quadrant boundaries only (Enabling x Upstream |
    # Safeguarding x Upstream | Enabling x Downstream | Safeguarding x
    # Downstream), matching Figure 1's column groupings.
    for _, _, end in TARGET_QUADRANT_RUNS[:-1]:
        ax.axvline(end - 0.5, color=_QUADRANT_LINE, linewidth=1.3, zorder=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Coded articles", fontsize=9, color=_INK_MUTED)
    cbar.ax.tick_params(labelsize=8, colors=_INK_MUTED)
    cbar.outline.set_visible(False)

    if title:
        n_total = int(data.sum())
        ax.set_title(f"{title}\n(n = {n_total:,} instrument-target entries)", fontsize=11.5, pad=12, color=_INK)

    fig.subplots_adjust(bottom=0.46, right=0.98, top=0.9)

    # --- Nested header bands below the rotated target tick labels --------
    # Level 1 (closest to the ticks): Enabling / Safeguarding, one segment
    # per contiguous run (there are two of each — Up- and Downstream).
    # Level 2 (outermost): UPSTREAM / DOWNSTREAM, one segment each.
    # Positioned from the actual rendered extent of the tick labels (which
    # varies with font/label length) rather than a fixed offset.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    min_y_axes = min(
        inv.transform((0, lbl.get_window_extent(renderer=renderer).y0))[1] for lbl in ax.get_xticklabels()
    )

    trans = ax.get_xaxis_transform()
    row_gap = 0.055
    row_height = 0.005
    label_pad = 0.018

    def _draw_band(runs: list[tuple[str, int, int]], y_line: float, *, bold: bool, upper: bool):
        label_y = y_line - label_pad
        for name, start, end in runs:
            ax.plot(
                [start - 0.5, end - 0.5], [y_line, y_line],
                color=_BAND_LINE, linewidth=1.1, transform=trans, clip_on=False,
            )
            # small end-ticks, like axis brackets
            for x in (start - 0.5, end - 0.5):
                ax.plot(
                    [x, x], [y_line, y_line + row_height], color=_BAND_LINE, linewidth=1.1,
                    transform=trans, clip_on=False,
                )
            ax.text(
                (start + end - 1) / 2, label_y, name.upper() if upper else name,
                transform=trans, ha="center", va="top", fontsize=9 if not bold else 9.5,
                fontweight="bold" if bold else "normal", color=_INK, clip_on=False,
            )

    line_y1 = min_y_axes - 0.03
    _draw_band(TARGET_FUNCTION_RUNS, line_y1, bold=False, upper=False)

    line_y2 = line_y1 - row_gap
    _draw_band(TARGET_LOCATION_RUNS, line_y2, bold=True, upper=True)

    return fig
