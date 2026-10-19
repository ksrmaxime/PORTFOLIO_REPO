"""Step LLM 1 — pertinence IA des articles repérés par mots-clés.

Entrée : sortie annotée du filtre par mots-clés (…_annotated.parquet).
Pour chaque article (level 6) contenant un mot-clé, le LLM (Apertus) indique si le terme
repéré fait référence à de l'IA : SPECIFIQUE / NEUTRE / NON.

Méthode : pas de génération libre. Pour chaque article, on mesure la probabilité que le
modèle réponde chacun des trois mots (une passe avant par réponse possible) ; la classe
retenue est la plus probable, et les probabilités sont gardées comme score de confiance.
Une justification d'une phrase est ensuite générée, conditionnée sur la classe retenue.

Sorties (--output_base) :
  <base>.parquet / .csv        le dataset d'entrée complet + colonnes AIREL_* (remplies sur
                               les articles traités, vides ailleurs)
  <base>_articles.csv          seulement les articles traités (lecture / vérification)
  <base>_eval.txt              si --gold : comparaison au codage manuel
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import math
import os
import time

import pandas as pd

from src import ai_relevance_prompts as P
from src.keyword_filter import KEYWORDS, compile_keywords, find_surface_terms


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def evaluate(out: pd.DataFrame, gold_path: str, report_path: str) -> None:
    g = read_any(gold_path)
    if "AI_RELEVANT" not in g.columns:
        print(f"[eval] pas de colonne AI_RELEVANT dans {gold_path} — évaluation sautée")
        return
    g = g[["law_id", "order_index", "AI_RELEVANT"]].copy()
    g["AI_RELEVANT"] = g["AI_RELEVANT"].astype(str).str.strip().str.upper()
    g = g[g["AI_RELEVANT"].isin(P.LABELS)]
    g["order_index"] = g["order_index"].astype(str)
    pred = out[out["AIREL_LABEL"].notna() & (out["AIREL_LABEL"] != "")][["law_id", "order_index", "AIREL_LABEL"]].copy()
    pred["order_index"] = pred["order_index"].astype(str)
    m = pred.merge(g, on=["law_id", "order_index"], how="inner")
    lines = [f"Articles codés à la main et traités : {len(m)}"]
    if len(m):
        acc = (m["AIREL_LABEL"] == m["AI_RELEVANT"]).mean()
        lines.append(f"Exactitude : {acc:.1%}")
        conf = pd.crosstab(m["AI_RELEVANT"], m["AIREL_LABEL"], rownames=["gold"], colnames=["LLM"]).reindex(
            index=P.LABELS, columns=P.LABELS, fill_value=0)
        lines.append("\nMatrice de confusion (lignes = codage manuel, colonnes = LLM) :\n" + conf.to_string())
        for lab in P.LABELS:
            tp = int(((m["AIREL_LABEL"] == lab) & (m["AI_RELEVANT"] == lab)).sum())
            npred = int((m["AIREL_LABEL"] == lab).sum())
            ngold = int((m["AI_RELEVANT"] == lab).sum())
            prec = tp / npred if npred else float("nan")
            rec = tp / ngold if ngold else float("nan")
            lines.append(f"{lab:10s} précision {prec:.1%}  rappel {rec:.1%}  (gold {ngold}, LLM {npred})")
    txt = "\n".join(lines)
    print("\n=== ÉVALUATION ===\n" + txt)
    Path(report_path).write_text(txt + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="…_annotated.parquet du filtre par mots-clés")
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4, help="articles par passe (x3 séquences)")
    ap.add_argument("--no_justification", action="store_true", help="ne pas générer de justification")
    ap.add_argument("--max_new_tokens", type=int, default=90)
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que N articles (test)")
    ap.add_argument("--gold", default=None, help="CSV de codage manuel (colonne AI_RELEVANT) pour évaluer")
    args = ap.parse_args()

    df = read_any(args.input).reset_index(drop=True)
    lvl = pd.to_numeric(df["level"], errors="coerce")
    hit = df["kw_hit"].astype(str).str.lower().isin(["true", "1"]) if df["kw_hit"].dtype != bool else df["kw_hit"]
    todo = df.index[(lvl == 6) & hit].tolist()
    if args.limit:
        todo = todo[: args.limit]
    print(f"Lu : {args.input} ({len(df):,} lignes) — articles à classer : {len(todo):,}")

    compiled = compile_keywords(KEYWORDS)
    contexts = P.build_contexts(df)

    prompts_user, terms_col = {}, {}
    for i in todo:
        row = df.loc[i]
        found = find_surface_terms(f"{row['label']}\n{row['text']}", compiled)
        terms = [f["term"] for f in found] or [t for t in str(row.get("kw_keywords", "")).split("|") if t]
        terms_col[i] = terms
        prompts_user[i] = P.build_user_prompt(row, contexts[i], terms)

    from src.client import LLMConfig, TransformersClient
    t0 = time.time()
    client = TransformersClient(LLMConfig(model_path=args.model_path, dtype=args.dtype,
                                          trust_remote_code=args.trust_remote_code))
    print(f"Modèle chargé en {time.time() - t0:.0f} s")

    chat_prompts = [client.build_chat_prompt(P.SYSTEM_PROMPT, prompts_user[i]) for i in todo]
    n_tok = [len(client.tok(p, add_special_tokens=False)["input_ids"]) for p in chat_prompts]
    print(f"Longueur des prompts (tokens) : médiane {sorted(n_tok)[len(n_tok)//2] if n_tok else 0}, max {max(n_tok) if n_tok else 0}")

    t0 = time.time()
    logps = client.score_options(chat_prompts, P.LABELS, batch_size=args.batch_size)
    print(f"Classification : {len(todo)} articles en {time.time() - t0:.0f} s")

    labels, probs = [], []
    for lp in logps:
        pr = softmax(lp)
        probs.append(pr)
        labels.append(P.LABELS[max(range(len(pr)), key=lambda k: pr[k])])

    justifs = [""] * len(todo)
    if not args.no_justification and todo:
        t0 = time.time()
        jprompts = [cp + lab + P.JUSTIF_SUFFIX for cp, lab in zip(chat_prompts, labels)]
        raw = client.generate_from_prompts(jprompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        justifs = [" ".join(r.split("\n\n")[0].split()) for r in raw]
        print(f"Justifications : {time.time() - t0:.0f} s")

    for c in ["AIREL_TERMS", "AIREL_LABEL", "AIREL_P_SPECIFIQUE", "AIREL_P_NEUTRE", "AIREL_P_NON",
              "AIREL_CONFIDENCE", "AIREL_JUSTIFICATION", "AIREL_PROMPT_TOKENS"]:
        df[c] = pd.NA
    for k, i in enumerate(todo):
        df.at[i, "AIREL_TERMS"] = " | ".join(terms_col[i])
        df.at[i, "AIREL_LABEL"] = labels[k]
        df.at[i, "AIREL_P_SPECIFIQUE"] = round(probs[k][0], 4)
        df.at[i, "AIREL_P_NEUTRE"] = round(probs[k][1], 4)
        df.at[i, "AIREL_P_NON"] = round(probs[k][2], 4)
        df.at[i, "AIREL_CONFIDENCE"] = round(max(probs[k]), 4)
        df.at[i, "AIREL_JUSTIFICATION"] = justifs[k]
        df.at[i, "AIREL_PROMPT_TOKENS"] = n_tok[k]

    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    base = Path(f"{args.output_base}_job{job_id}")
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{base}.parquet", index=False)
    df.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
    art_cols = [c for c in ["law_id", "sr_number", "doc_category", "order_index", "label", "text", "kw_keywords", "kw_tier"] if c in df.columns]
    art = df.loc[todo, art_cols + [c for c in df.columns if c.startswith("AIREL_")]]
    art.insert(0, "act_title", [contexts[i]["act_title"] for i in todo])
    art.to_csv(f"{base}_articles.csv", index=False, encoding="utf-8-sig")
    # exemple de prompt complet, pour relecture
    if todo:
        Path(f"{base}_prompt_example.txt").write_text(chat_prompts[0], encoding="utf-8")

    print("\n=== BILAN ===")
    vc = pd.Series(labels).value_counts()
    for lab in P.LABELS:
        print(f"{lab:10s} {int(vc.get(lab, 0)):4d}")
    conf = pd.Series([max(p) for p in probs])
    if len(conf):
        print(f"Confiance : médiane {conf.median():.2f} ; {int((conf < 0.6).sum())} article(s) sous 0.60 (à relire en priorité)")
    if "kw_tier" in df.columns and todo:
        print("\nPar niveau de mot-clé :")
        print(pd.crosstab(df.loc[todo, "kw_tier"], pd.Series(labels, index=todo)).to_string())
    print(f"\nSaved: {base}.parquet / .csv, {base}_articles.csv, {base}_prompt_example.txt")

    if args.gold:
        evaluate(df, args.gold, f"{base}_eval.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
