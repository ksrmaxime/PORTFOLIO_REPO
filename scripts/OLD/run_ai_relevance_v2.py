"""Step LLM 1 (v2) — pertinence IA des articles repérés par mots-clés.

Entrée : sortie annotée du filtre par mots-clés (…_annotated.parquet).
Pour chaque article (level 6) contenant un mot-clé :
  1. RÈGLE : si l'IA y est nommée explicitement (P.RULE_SPECIFIQUE_KEYWORDS), l'article est
     classé SPECIFIQUE sans LLM (AIREL_METHOD = "regle").
  2. LLM sinon : Apertus répond par une lettre A/B/C (= SPECIFIQUE/NEUTRE/NON). On ne génère
     pas la réponse : on lit en une passe la probabilité des trois lettres comme prochain token.
     Calibration contextuelle : la préférence du modèle pour chaque lettre, mesurée sur des
     prompts sans contenu (P.CONTENT_FREE_INPUTS), est retirée des probabilités. Les deux
     versions (brute / calibrée) sont gardées ; le label retenu est calibré sauf --no_calibration.
     Une justification d'une phrase est ensuite générée, conditionnée sur la lettre retenue.

Sorties (--output_base, suffixe _job<SLURM_JOB_ID>) :
  .parquet / .csv          dataset d'entrée complet + colonnes AIREL_*
  _articles.csv            seulement les articles traités
  _prompt_example.txt      un prompt complet, pour relecture
  _eval.txt                si --gold : comparaison au gold
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


def calibrate(p, p_cf):
    q = [a / b for a, b in zip(p, p_cf)]
    s = sum(q)
    return [v / s for v in q]


def _report(m: pd.DataFrame, pred_col: str, title: str) -> list:
    lines = [f"\n--- {title} : {len(m)} articles ---"]
    if not len(m):
        return lines
    acc = (m[pred_col] == m["AI_RELEVANT"]).mean()
    base = m["AI_RELEVANT"].value_counts(normalize=True).max()
    lines.append(f"Exactitude : {acc:.1%}  (référence « toujours la classe majoritaire » : {base:.1%})")
    conf = pd.crosstab(m["AI_RELEVANT"], m[pred_col], rownames=["gold"], colnames=["prédit"]).reindex(
        index=P.LABELS, columns=P.LABELS, fill_value=0)
    lines.append(conf.to_string())
    for lab in P.LABELS:
        tp = int(((m[pred_col] == lab) & (m["AI_RELEVANT"] == lab)).sum())
        npred, ngold = int((m[pred_col] == lab).sum()), int((m["AI_RELEVANT"] == lab).sum())
        prec = f"{tp / npred:.1%}" if npred else "  n/a"
        rec = f"{tp / ngold:.1%}" if ngold else "  n/a"
        lines.append(f"{lab:10s} précision {prec:>6s}  rappel {rec:>6s}  (gold {ngold}, prédit {npred})")
    return lines


def evaluate(out: pd.DataFrame, gold_path: str, report_path: str) -> None:
    g = read_any(gold_path)
    if "AI_RELEVANT" not in g.columns:
        print(f"[eval] pas de colonne AI_RELEVANT dans {gold_path} — évaluation sautée")
        return
    g = g[["law_id", "order_index", "AI_RELEVANT"]].copy()
    g["AI_RELEVANT"] = g["AI_RELEVANT"].astype(str).str.strip().str.upper()
    g = g[g["AI_RELEVANT"].isin(P.LABELS)]
    g["order_index"] = g["order_index"].astype(str)
    cols = ["law_id", "order_index", "AIREL_LABEL", "AIREL_LABEL_BRUT", "AIREL_METHOD"]
    pred = out[out["AIREL_LABEL"].notna() & (out["AIREL_LABEL"] != "")][cols].copy()
    pred["order_index"] = pred["order_index"].astype(str)
    m = pred.merge(g, on=["law_id", "order_index"], how="inner")
    llm = m[m["AIREL_METHOD"] == "llm"]
    lines = [f"Articles traités et présents dans le gold : {len(m)} (règle {len(m) - len(llm)}, LLM {len(llm)})"]
    lines += _report(m, "AIREL_LABEL", "Pipeline complète (règle + LLM)")
    lines += _report(llm, "AIREL_LABEL", "LLM seul, probabilités calibrées")
    lines += _report(llm, "AIREL_LABEL_BRUT", "LLM seul, probabilités brutes")
    rule = m[m["AIREL_METHOD"] == "regle"]
    if len(rule):
        lines.append(f"\nRègle « IA nommée » : {int((rule['AI_RELEVANT'] == 'SPECIFIQUE').sum())}/{len(rule)} SPECIFIQUE dans le gold")
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
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--no_calibration", action="store_true", help="label = probabilités brutes")
    ap.add_argument("--no_rule", action="store_true", help="envoyer aussi au LLM les articles où l'IA est nommée")
    ap.add_argument("--no_justification", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=90)
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que N articles (test)")
    ap.add_argument("--gold", default=None, help="CSV gold (colonne AI_RELEVANT) pour évaluer")
    args = ap.parse_args()

    df = read_any(args.input).reset_index(drop=True)
    lvl = pd.to_numeric(df["level"], errors="coerce")
    hit = df["kw_hit"].astype(str).str.lower().isin(["true", "1"]) if df["kw_hit"].dtype != bool else df["kw_hit"]
    todo = df.index[(lvl == 6) & hit].tolist()
    if args.limit:
        todo = todo[: args.limit]

    rule_set = set(P.RULE_SPECIFIQUE_KEYWORDS)
    def is_rule(i):
        return (not args.no_rule) and bool(rule_set & set(str(df.at[i, "kw_keywords"]).split("|")))
    rule_idx = [i for i in todo if is_rule(i)]
    llm_idx = [i for i in todo if not is_rule(i)]
    print(f"Lu : {args.input} ({len(df):,} lignes) — articles à classer : {len(todo):,} "
          f"(règle « IA nommée » : {len(rule_idx)}, LLM : {len(llm_idx)})")

    compiled = compile_keywords(KEYWORDS)
    contexts = P.build_contexts(df)
    terms_col, prompts_user = {}, {}
    for i in todo:
        row = df.loc[i]
        found = find_surface_terms(f"{row['label']}\n{row['text']}", compiled)
        terms_col[i] = [f["term"] for f in found] or [t for t in str(row.get("kw_keywords", "")).split("|") if t]
        prompts_user[i] = P.build_user_prompt(row, contexts[i], terms_col[i])

    res = {}   # i -> dict de colonnes AIREL_*
    for i in rule_idx:
        explicit = [k for k in str(df.at[i, "kw_keywords"]).split("|") if k in rule_set]
        res[i] = {"AIREL_METHOD": "regle", "AIREL_LABEL": "SPECIFIQUE", "AIREL_LABEL_BRUT": "SPECIFIQUE",
                  "AIREL_JUSTIFICATION": "IA nommée explicitement (mot-clé : " + ", ".join(explicit) + ")"}

    chat_prompts, n_tok = [], []
    if llm_idx:
        from src.client import LLMConfig, TransformersClient
        t0 = time.time()
        client = TransformersClient(LLMConfig(model_path=args.model_path, dtype=args.dtype,
                                              trust_remote_code=args.trust_remote_code))
        print(f"Modèle chargé en {time.time() - t0:.0f} s")

        print("Tokenisation des réponses :")
        for o in client.option_first_tokens(P.LETTERS):
            flag = "OK" if len(o["ids"]) == 1 else "ATTENTION : plusieurs tokens, seul le premier est lu"
            print(f"  {o['option']!r} -> {o['tokens']} {o['ids']}  {flag}")
        if len({client.tok(l, add_special_tokens=False)['input_ids'][0] for l in P.LETTERS}) < len(P.LETTERS):
            raise RuntimeError("Deux lettres partagent le même premier token")

        cf = [client.build_chat_prompt(P.SYSTEM_PROMPT, u) for u in P.content_free_prompts()]
        cf_probs = [softmax(lp) for lp in client.next_token_logprobs(cf, P.LETTERS, batch_size=len(cf))]
        p_cf = [sum(p[k] for p in cf_probs) / len(cf_probs) for k in range(len(P.LETTERS))]
        print("Préférence de base du modèle (prompts vides) : " +
              ", ".join(f"{l}={v:.3f}" for l, v in zip(P.LETTERS, p_cf)))

        chat_prompts = [client.build_chat_prompt(P.SYSTEM_PROMPT, prompts_user[i]) for i in llm_idx]
        n_tok = [len(client.tok(p, add_special_tokens=False)["input_ids"]) for p in chat_prompts]
        print(f"Longueur des prompts (tokens) : médiane {sorted(n_tok)[len(n_tok)//2]}, max {max(n_tok)}")

        t0 = time.time()
        logps = client.next_token_logprobs(chat_prompts, P.LETTERS, batch_size=args.batch_size)
        print(f"Classification : {len(llm_idx)} articles en {time.time() - t0:.0f} s")

        letters = []
        for k, (i, lp) in enumerate(zip(llm_idx, logps)):
            raw = softmax(lp)
            cal = calibrate(raw, p_cf)
            use = raw if args.no_calibration else cal
            best = max(range(3), key=lambda j: use[j])
            letters.append(P.LETTERS[best])
            res[i] = {"AIREL_METHOD": "llm",
                      "AIREL_LABEL": P.LABELS[best],
                      "AIREL_LABEL_BRUT": P.LABELS[max(range(3), key=lambda j: raw[j])],
                      "AIREL_P_SPECIFIQUE": round(cal[0], 4), "AIREL_P_NEUTRE": round(cal[1], 4), "AIREL_P_NON": round(cal[2], 4),
                      "AIREL_P_SPECIFIQUE_BRUT": round(raw[0], 4), "AIREL_P_NEUTRE_BRUT": round(raw[1], 4), "AIREL_P_NON_BRUT": round(raw[2], 4),
                      "AIREL_CONFIDENCE": round(max(use), 4),
                      "AIREL_PROMPT_TOKENS": n_tok[k]}

        if not args.no_justification:
            t0 = time.time()
            jprompts = [cp + l + P.JUSTIF_SUFFIX for cp, l in zip(chat_prompts, letters)]
            raw_j = client.generate_from_prompts(jprompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            for i, r in zip(llm_idx, raw_j):
                res[i]["AIREL_JUSTIFICATION"] = " ".join(r.split("\n\n")[0].split())
            print(f"Justifications : {time.time() - t0:.0f} s")

    cols = ["AIREL_METHOD", "AIREL_TERMS", "AIREL_LABEL", "AIREL_LABEL_BRUT",
            "AIREL_P_SPECIFIQUE", "AIREL_P_NEUTRE", "AIREL_P_NON",
            "AIREL_P_SPECIFIQUE_BRUT", "AIREL_P_NEUTRE_BRUT", "AIREL_P_NON_BRUT",
            "AIREL_CONFIDENCE", "AIREL_JUSTIFICATION", "AIREL_PROMPT_TOKENS"]
    for c in cols:
        df[c] = pd.NA
    for i in todo:
        df.at[i, "AIREL_TERMS"] = " | ".join(terms_col[i])
        for c, v in res[i].items():
            df.at[i, c] = v

    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    base = Path(f"{args.output_base}_job{job_id}")
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{base}.parquet", index=False)
    df.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
    art_cols = [c for c in ["law_id", "sr_number", "doc_category", "order_index", "label", "text", "kw_keywords", "kw_tier"] if c in df.columns]
    art = df.loc[todo, art_cols + cols]
    art.insert(0, "act_title", [contexts[i]["act_title"] for i in todo])
    art.to_csv(f"{base}_articles.csv", index=False, encoding="utf-8-sig")
    if chat_prompts:
        Path(f"{base}_prompt_example.txt").write_text(chat_prompts[0], encoding="utf-8")

    print("\n=== BILAN ===")
    vc = df.loc[todo, "AIREL_LABEL"].value_counts()
    for lab in P.LABELS:
        print(f"{lab:10s} {int(vc.get(lab, 0)):4d}")
    if llm_idx:
        conf = df.loc[llm_idx, "AIREL_CONFIDENCE"].astype(float)
        print(f"LLM — confiance : médiane {conf.median():.2f} ; {int((conf < 0.6).sum())} article(s) sous 0.60")
        moved = int((df.loc[llm_idx, "AIREL_LABEL"] != df.loc[llm_idx, "AIREL_LABEL_BRUT"]).sum())
        print(f"Articles dont le label change avec la calibration : {moved}")
    print(f"\nSaved: {base}.parquet / .csv, {base}_articles.csv")

    if args.gold:
        evaluate(df, args.gold, f"{base}_eval.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
