from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # Remonte jusqu’au dossier qui contient pyproject.toml
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not find project root (pyproject.toml not found).")


def data_dir(kind: str) -> Path:
    root = project_root()
    if kind not in {"raw", "processed", "external"}:
        raise ValueError(f"Unknown data dir kind: {kind}")
    return root / "data" / kind


def output_dir(kind: str) -> Path:
    root = project_root()
    if kind not in {"plots", "graphs", "tables"}:
        raise ValueError(f"Unknown output dir kind: {kind}")
    return root / "output" / kind


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
