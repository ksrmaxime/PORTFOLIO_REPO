from __future__ import annotations

import pandas as pd

from portfolio_repo.fedlex.downloader import DownloadConfig, download_cc_xml_batch
from portfolio_repo.paths import data_dir, ensure_dir


def main() -> int:
    cat_path = data_dir("processed") / "fedlex" / "cc_catalog_fr_latest.csv"
    df = pd.read_csv(cat_path)

    log = download_cc_xml_batch(df, DownloadConfig(overwrite=False))

    out_dir = ensure_dir(data_dir("processed") / "fedlex")
    log_path = out_dir / "cc_download_log_fr.csv"
    log.to_csv(log_path, index=False)

    ok = int(log["ok"].sum())
    print(f"Downloaded OK: {ok}/{len(log)}. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
