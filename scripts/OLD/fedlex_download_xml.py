# scripts/fedlex_download_xml.py
from __future__ import annotations

import argparse

import pandas as pd
import requests

from portfolio_repo.paths import project_paths
from portfolio_repo.fedlex.download import now_iso, save_xml, safe_law_folder_name, sha256_bytes
from portfolio_repo.fedlex.manifestation import find_xml_manifestations


def looks_like_xml(content: bytes, content_type: str | None) -> bool:
    ct = (content_type or "").lower()
    if "xml" in ct:
        return True
    head = content[:200].lstrip()
    if head.startswith(b"<?xml"):
        return True
    # Akoma Ntoso often appears early
    if b"<akoma" in head.lower() or b"akoma" in head.lower():
        return True
    return False


def download_first_xml(urls: list[str]) -> tuple[bytes, int, str, str | None]:
    last_err: Exception | None = None
    for u in urls:
        try:
            r = requests.get(
                u,
                headers={"User-Agent": "portfolio_repo/0.1 (research)"},
                timeout=60,
                allow_redirects=True,
            )
            r.raise_for_status()
            ct = r.headers.get("Content-Type")
            if looks_like_xml(r.content, ct):
                return r.content, r.status_code, u, ct
            last_err = RuntimeError(f"URL did not return XML (Content-Type={ct})")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No candidate URL returned XML. Last error: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--registry", default="fedlex_registry.parquet")
    ap.add_argument("--langs", default="de,fr,it", help="Comma-separated languages to attempt")
    ap.add_argument("--max-laws", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="fedlex_download_log.parquet")
    args = ap.parse_args()

    paths = project_paths(args.repo_root)
    raw_dir = paths.data_raw

    reg_path = paths.data_processed / args.registry
    df = pd.read_parquet(reg_path)

    if "eli_uri" not in df.columns:
        raise RuntimeError(f"Registry {reg_path} has no 'eli_uri' column.")

    if args.max_laws is not None:
        df = df.head(args.max_laws)

    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    results = []

    for _, row in df.iterrows():
        cc_uri = str(row["eli_uri"])

        for lang in langs:
            folder = raw_dir / "fedlex_xml" / safe_law_folder_name(cc_uri)
            target = folder / f"{lang}.xml"

            if args.resume and target.exists():
                results.append(
                    {
                        "law_id": cc_uri,  # stable id
                        "eli_uri": cc_uri,
                        "lang": lang,
                        "ok": True,
                        "http_status": None,
                        "error": None,
                        "xml_url": None,
                        "xml_path": str(target),
                        "sha256": sha256_bytes(target.read_bytes()),
                        "downloaded_at_iso": None,
                        "skipped_existing": True,
                        "content_type": None,
                    }
                )
                continue

            try:
                urls = find_xml_manifestations(cc_uri, lang=lang)
                if not urls:
                    raise RuntimeError(f"No XML manifestation found via SPARQL for lang={lang}")

                xml_bytes, status, xml_url, ct = download_first_xml(urls)
                xml_path = save_xml(raw_dir, eli_uri=cc_uri, lang=lang, xml_bytes=xml_bytes)

                results.append(
                    {
                        "law_id": cc_uri,
                        "eli_uri": cc_uri,
                        "lang": lang,
                        "ok": True,
                        "http_status": status,
                        "error": None,
                        "xml_url": xml_url,
                        "xml_path": str(xml_path),
                        "sha256": sha256_bytes(xml_bytes),
                        "downloaded_at_iso": now_iso(),
                        "skipped_existing": False,
                        "content_type": ct,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "law_id": cc_uri,
                        "eli_uri": cc_uri,
                        "lang": lang,
                        "ok": False,
                        "http_status": None,
                        "error": str(e),
                        "xml_url": None,
                        "xml_path": None,
                        "sha256": None,
                        "downloaded_at_iso": now_iso(),
                        "skipped_existing": False,
                        "content_type": None,
                    }
                )

    out_df = pd.DataFrame(results)
    out_path = paths.data_processed / args.out
    out_df.to_parquet(out_path, index=False)

    print(f"Wrote download log: {out_path} ({len(out_df)} rows)")
    print(f"OK: {(out_df['ok'] == True).sum()} | FAIL: {(out_df['ok'] == False).sum()}")

    if (out_df["ok"] == False).any():
        print("\nFirst failures:")
        print(out_df.loc[out_df["ok"] == False, ["eli_uri", "lang", "error"]].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


