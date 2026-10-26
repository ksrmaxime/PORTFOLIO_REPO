"""Step LLM 2 — cible (cadran + sous-catégorie) et instrument des articles SPECIFIQUE.

Entrée : sortie de l'étape 1 (ai_relevance_job<ID>.parquet/.csv). Seuls les articles avec
AIREL_LABEL == SPECIFIQUE sont classés.
Trois questions par article (src/policy_classification_prompts.py), réponse en une lettre lue
par probabilité (une passe avant par question) :
  Q1 cadran (ou E = aucun mécanisme) -> Q2 sous-catégorie du cadran -> Q3 instrument.
Q2 et Q3 sont sautées pour les articles « aucun mécanisme ».
Une justification d'une phrase est générée pour la cible et pour l'instrument.

Sorties (--output_base, suffixe _job<SLURM_JOB_ID>) :
  .parquet / .csv     dataset d'entrée + colonnes POL_*
  _articles.csv       seulement les articles classés
  _prompt_example_q1/q2/q3.txt
  _eval.txt           si --gold (colonnes TARGET / SUBTARGET / INSTRUMENT)
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

from src import ai_relevance_prompts as A
from src import policy_classification_prompts as Q
from src.keyword_filter import KEYWORDS, compile_keywords, find_in_text, find_surface_terms


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def ranked(probs, codes):
    order = sorted(range(len(probs)), key=lambda j: -probs[j])
    return [(codes[j], probs[j]) for j in order]


def check_letters(client, letters):
    print("Tokenisation des réponses :")
    firsts = set()
    for o in client.option_first_tokens(letters):
        flag = "OK" if len(o["ids"]) == 1 else "ATTENTION : plusieurs tokens, seul le premier est lu"
        print(f"  {o['option']!r} -> {o['tokens']} {o['ids']}  {flag}")
        firsts.add(o["ids"][0])
    if len(firsts) < len(letters):
        raise RuntimeError("Deux lettres partagent le même premier token")


# ------------------------------------------------------------------ évaluation
def _split(v):
    return [x for x in str(v or "").split("|") if x]


def evaluate(out: pd.DataFrame, gold_path: str, report_path: str) -> None:
    g = read_any(gold_path)
    need = {"TARGET", "SUBTARGET", "INSTRUMENT", "AI_RELEVANT"}
    if not need <= set(g.columns):
        print(f"[eval] colonnes {need - set(g.columns)} absentes de {gold_path} — évaluation sautée")
        return
    g = g[g["AI_RELEVANT"].astype(str).str.upper() == "SPECIFIQUE"][["law_id", "order_index", "TARGET", "SUBTARGET", "INSTRUMENT"]].copy()
    g["order_index"] = g["order_index"].astype(str)
    pred = out[out["POL_QUADRANT"].notna() & (out["POL_QUADRANT"] != "")].copy()
    pred["order_index"] = pred["order_index"].astype(str)
    m = pred.merge(g, on=["law_id", "order_index"], how="inner")
    for c in [c for c in m.columns if c.startswith("POL_")]:
        m[c] = m[c].astype(object).where(m[c].notna(), "").astype(str)
    lines = [f"Articles classés et SPECIFIQUE dans le gold : {len(m)} (sur {len(pred)} classés, {len(g)} SPECIFIQUE dans le gold)"]
    if not len(m):
        Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return

    gt = m["TARGET"].map(_split)
    gs = m["SUBTARGET"].map(_split)
    gi = m["INSTRUMENT"].map(_split)
    pt = m["POL_TARGET"].fillna("")
    ok_t = [(p in t) if t else (p == "") for p, t in zip(pt, gt)]
    ok_t2 = [ok or (str(p2) in t) for ok, p2, t in zip(ok_t, m["POL_TARGET_2"].fillna(""), gt)]
    lines.append(f"\nCadran : exactitude {sum(ok_t) / len(m):.1%}   (vrai dans les 2 premières réponses : {sum(ok_t2) / len(m):.1%})")
    gt1 = [t[0] if t else "(aucun)" for t in gt]
    lines.append(pd.crosstab(pd.Series(gt1, name="gold"), pt.replace("", "(aucun)").rename("prédit")).to_string())

    mech = [bool(t) for t in gt]
    both = [i for i, (mk, ok) in enumerate(zip(mech, ok_t)) if mk and ok]
    ok_s = [m["POL_SUBTARGET"].iloc[i] in gs.iloc[i] for i in both]
    lines.append(f"\nSous-catégorie, quand le cadran est juste ({len(both)} articles) : exactitude {sum(ok_s) / max(1, len(both)):.1%}")
    ok_ts = [(pt.iloc[i] in gt.iloc[i]) and (m["POL_SUBTARGET"].iloc[i] in gs.iloc[i]) for i in range(len(m)) if mech[i]]
    lines.append(f"Cadran + sous-catégorie justes (articles avec mécanisme, {sum(mech)}) : {sum(ok_ts) / max(1, sum(mech)):.1%}")
    sub_g = [gs.iloc[i][0] for i in range(len(m)) if mech[i] and gs.iloc[i]]
    sub_p = [m["POL_SUBTARGET"].iloc[i] or "(aucun)" for i in range(len(m)) if mech[i] and gs.iloc[i]]
    lines.append(pd.crosstab(pd.Series(sub_g, name="gold"), pd.Series(sub_p, name="prédit")).to_string())

    idx = [i for i in range(len(m)) if mech[i]]
    ok_i = [m["POL_INSTRUMENT"].iloc[i] in gi.iloc[i] for i in idx]
    ok_i2 = [ok or (m["POL_INSTRUMENT_2"].iloc[i] in gi.iloc[i]) for ok, i in zip(ok_i, idx)]
    lines.append(f"\nInstrument (articles avec mécanisme, {len(idx)}) : exactitude {sum(ok_i) / max(1, len(idx)):.1%}"
                 f"   (vrai dans les 2 premières : {sum(ok_i2) / max(1, len(idx)):.1%})")
    lines.append(pd.crosstab(pd.Series([gi.iloc[i][0] for i in idx], name="gold"),
                             pd.Series([m["POL_INSTRUMENT"].iloc[i] or "(aucun)" for i in idx], name="prédit")).to_string())
    txt = "\n".join(lines)
    print("\n=== ÉVALUATION ===\n" + txt)
    Path(report_path).write_text(txt + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="sortie de l'étape 1 (ai_relevance_job<ID>.parquet/.csv)")
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--no_definitions", action="store_true")
    ap.add_argument("--no_justification", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=90)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gold", default=None, help="gold avec colonnes TARGET / SUBTARGET / INSTRUMENT")
    ap.add_argument("--select_from_gold", action="store_true",
                    help="classer les articles SPECIFIQUE du GOLD (au lieu de ceux de l'étape 1) : "
                         "évalue l'étape 2 seule, sans les erreurs de l'étape 1")
    args = ap.parse_args()

    df = read_any(args.input).reset_index(drop=True)
    if "AIREL_LABEL" not in df.columns:
        raise RuntimeError("Pas de colonne AIREL_LABEL : l'entrée doit être la sortie de l'étape 1")
    lvl = pd.to_numeric(df["level"], errors="coerce")
    if args.select_from_gold:
        if not args.gold:
            raise RuntimeError("--select_from_gold demande --gold")
        g = read_any(args.gold)
        keys = set(zip(g.loc[g["AI_RELEVANT"].astype(str).str.upper() == "SPECIFIQUE", "law_id"].astype(str),
                       g.loc[g["AI_RELEVANT"].astype(str).str.upper() == "SPECIFIQUE", "order_index"].astype(str)))
        sel = [(str(a), str(b)) in keys for a, b in zip(df["law_id"], df["order_index"])]
        todo = df.index[(lvl == 6) & pd.Series(sel, index=df.index)].tolist()
        print("Sélection : articles SPECIFIQUE du gold")
    else:
        todo = df.index[(lvl == 6) & (df["AIREL_LABEL"].astype(str) == "SPECIFIQUE")].tolist()
    if args.limit:
        todo = todo[: args.limit]
    print(f"Lu : {args.input} ({len(df):,} lignes) — articles SPECIFIQUE à classer : {len(todo)}")

    compiled = compile_keywords(KEYWORDS)
    contexts = A.build_contexts(df)
    def_index = {} if args.no_definitions else A.find_definition_articles(df)
    base_args = {}
    for i in todo:
        row = df.loc[i]
        full = f"{row['label']}\n{row['text']}"
        terms = [f["term"] for f in find_surface_terms(full, compiled)]
        defs = ""
        if def_index:
            kw_here = set(find_in_text(full, compiled))
            sub = [c for c in compiled if c.kw in kw_here]
            defs = A.definitions_for(df, i, def_index, sub, find_in_text) if sub else ""
        base_args[i] = (row, contexts[i], terms, defs)

    from src.client import LLMConfig, TransformersClient
    t0 = time.time()
    client = TransformersClient(LLMConfig(model_path=args.model_path, dtype=args.dtype,
                                          trust_remote_code=args.trust_remote_code))
    print(f"Modèle chargé en {time.time() - t0:.0f} s")
    check_letters(client, Q.LETTERS)

    res = {i: {} for i in todo}
    examples = {}

    # ---- Q1 cadran
    sys1, let1 = Q.q1_system()
    codes1 = [c for c, _ in Q.QUADRANTS]
    p1 = [client.build_chat_prompt(sys1, Q.build_user_prompt(*base_args[i], "q1", let1)) for i in todo]
    t0 = time.time()
    lp1 = client.next_token_logprobs(p1, let1, batch_size=args.batch_size)
    print(f"Q1 cadran : {len(todo)} articles en {time.time() - t0:.0f} s")
    letter1 = {}
    for i, prompt, lp in zip(todo, p1, lp1):
        r = ranked(softmax(lp), codes1)
        res[i].update({"POL_QUADRANT": r[0][0] or "AUCUN", "POL_TARGET": r[0][0], "POL_TARGET_P": round(r[0][1], 4),
                       "POL_TARGET_2": r[1][0] or "AUCUN", "POL_TARGET_2_P": round(r[1][1], 4)})
        letter1[i] = let1[codes1.index(r[0][0])]
        examples.setdefault("q1", prompt)
    mech = [i for i in todo if res[i]["POL_TARGET"]]

    # ---- Q2 sous-catégorie (prompt système selon le cadran)
    p2, codes2, let2 = {}, {}, {}
    for i in mech:
        sys2, l2, subs = Q.q2_system(res[i]["POL_TARGET"])
        p2[i] = client.build_chat_prompt(sys2, Q.build_user_prompt(*base_args[i], "q2", l2))
        codes2[i], let2[i] = subs, l2
        examples.setdefault("q2", p2[i])
    if mech:
        t0 = time.time()
        lp2 = client.next_token_logprobs([p2[i] for i in mech], Q.LETTERS[:5], batch_size=args.batch_size)
        print(f"Q2 sous-catégorie : {len(mech)} articles en {time.time() - t0:.0f} s")
        for i, lp in zip(mech, lp2):
            r = ranked(softmax(lp), codes2[i])
            res[i].update({"POL_SUBTARGET": r[0][0], "POL_SUBTARGET_P": round(r[0][1], 4),
                           "POL_SUBTARGET_2": r[1][0], "POL_SUBTARGET_2_P": round(r[1][1], 4)})

    # ---- Q3 instrument
    sys3, let3 = Q.q3_system()
    codes3 = [c for c, _ in Q.INSTRUMENTS]
    p3 = {i: client.build_chat_prompt(sys3, Q.build_user_prompt(*base_args[i], "q3", let3)) for i in mech}
    letter3 = {}
    if mech:
        examples["q3"] = p3[mech[0]]
        t0 = time.time()
        lp3 = client.next_token_logprobs([p3[i] for i in mech], let3, batch_size=args.batch_size)
        print(f"Q3 instrument : {len(mech)} articles en {time.time() - t0:.0f} s")
        for i, lp in zip(mech, lp3):
            r = ranked(softmax(lp), codes3)
            res[i].update({"POL_INSTRUMENT": r[0][0], "POL_INSTRUMENT_P": round(r[0][1], 4),
                           "POL_INSTRUMENT_2": r[1][0], "POL_INSTRUMENT_2_P": round(r[1][1], 4)})
            letter3[i] = let3[codes3.index(r[0][0])]

    # ---- justifications
    if not args.no_justification and todo:
        t0 = time.time()
        jt = client.generate_from_prompts([p + letter1[i] + Q.JUSTIF_SUFFIX for i, p in zip(todo, p1)],
                                          max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        for i, r in zip(todo, jt):
            res[i]["POL_TARGET_JUSTIFICATION"] = " ".join(r.split("\n\n")[0].split())
        if mech:
            ji = client.generate_from_prompts([p3[i] + letter3[i] + Q.JUSTIF_SUFFIX for i in mech],
                                              max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            for i, r in zip(mech, ji):
                res[i]["POL_INSTRUMENT_JUSTIFICATION"] = " ".join(r.split("\n\n")[0].split())
        print(f"Justifications : {time.time() - t0:.0f} s")

    cols = ["POL_QUADRANT", "POL_TARGET", "POL_TARGET_P", "POL_TARGET_2", "POL_TARGET_2_P",
            "POL_SUBTARGET", "POL_SUBTARGET_P", "POL_SUBTARGET_2", "POL_SUBTARGET_2_P",
            "POL_INSTRUMENT", "POL_INSTRUMENT_P", "POL_INSTRUMENT_2", "POL_INSTRUMENT_2_P",
            "POL_DEFINITION", "POL_TARGET_JUSTIFICATION", "POL_INSTRUMENT_JUSTIFICATION"]
    for c in cols:
        df[c] = pd.NA
    for i in todo:
        df.at[i, "POL_DEFINITION"] = base_args[i][3].strip()
        for c, v in res[i].items():
            df.at[i, c] = v

    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    base = Path(f"{args.output_base}_job{job_id}")
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{base}.parquet", index=False)
    df.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
    art_cols = [c for c in ["law_id", "sr_number", "order_index", "label", "text", "AIREL_TERMS", "AIREL_METHOD"] if c in df.columns]
    art = df.loc[todo, art_cols + cols]
    art.insert(0, "act_title", [contexts[i]["act_title"] for i in todo])
    art.to_csv(f"{base}_articles.csv", index=False, encoding="utf-8-sig")
    for q, p in examples.items():
        Path(f"{base}_prompt_example_{q}.txt").write_text(p, encoding="utf-8")

    print("\n=== BILAN ===")
    sub = df.loc[todo]
    print(pd.crosstab(sub["POL_QUADRANT"], sub["POL_SUBTARGET"].fillna("-")).to_string())
    print("\nInstruments :\n" + sub["POL_INSTRUMENT"].value_counts().to_string())
    print(f"\nSaved: {base}.parquet / .csv, {base}_articles.csv")

    if args.gold:
        evaluate(df, args.gold, f"{base}_eval.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
