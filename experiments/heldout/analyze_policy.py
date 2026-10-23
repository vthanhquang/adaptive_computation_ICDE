"""Policy analysis on the TEST traces of a frozen run.

  pricing   : predicted DB ms of VALUE_PROBE / EXEC_BOUNDED (from visible pre-probe features incl. EXPLAIN,
              fitted on TRAIN) vs measured ms on every test candidate: Spearman rho, AUROC(>100 ms), MAE(log)
  choices   : which probe the adaptive policy buys in which state, from the chosen operating point:
              P(VALUE_PROBE on c0 | c0 has literal predicates), P(EXEC on c0), P(EXEC on a new candidate |
              >=2 candidates at that time), probes per question, and the same for the fixed bundle

Usage:  python -m experiments.heldout.analyze_policy pg|sqlite|pg_x10|pg_x100 [--price 1.0] [--budget 2.0]
Output: <run>/policy_analysis_p<price>_B<budget>.json
"""
import argparse, json, pickle
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from plangate.evaluation.heldout import chosen_traces, load, run_dir, run_meta, run_scale
from plangate.features import sql_features
from plangate.planning import adaptive as AD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--price", default="1.0"); ap.add_argument("--budget", default="2.0")
    a = ap.parse_args()
    d = run_dir(a.run)
    names = list(run_meta(a.run)["splits"])
    scale = run_scale(a.run)
    test, _, _ = load(names[-1], scale)
    pl = pickle.loads((d / "planner.pkl").read_bytes())
    out = {"eval_dir": a.run, "scale": scale}
    # ---- pricing quality
    pr = {}
    for op in (AD.VP, AD.EX):
        p, y = [], []
        for r in test:
            ents = [r[k] for k in ("c0", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN") if r.get(k)]
            for e in ents:
                p.append(pl.pred_ms(e, op)); y.append(e["db_ms"].get(op, 0.0))
        p, y = np.array(p), np.array(y)
        rho = spearmanr(p, y).correlation
        lab = (y > 100).astype(int)
        auc = roc_auc_score(lab, p) if 0 < lab.sum() < len(lab) else float("nan")
        pr[op] = {"n": len(y), "spearman": float(rho), "auroc_gt100ms": float(auc), "share_gt100ms": float(lab.mean()),
                  "mae_log": float(np.mean(np.abs(np.log1p(p) - np.log1p(y)))), "mean_ms": float(y.mean()),
                  "p95_ms": float(np.percentile(y, 95))}
    out["pricing"] = pr
    # ---- probe choices at the chosen operating point
    T = {r["question_id"]: r for r in test}
    pref = f"p{float(a.price)}_B{float(a.budget)}_"
    traces = defaultdict(list)
    for x in chosen_traces(a.run):
        if x["config"].startswith(pref):
            traces[x["config"][len(pref):]].append(x)
    ch = {}
    for pol, outs in traces.items():
        c = defaultdict(lambda: [0, 0])
        probes = []
        for o in outs:
            r = T[o["q"]]
            lit = sql_features(r["c0"]["sql"])["sql_literals"] > 0
            n_c, vp0, ex0, ex_new, n_p = 1, False, False, False, 0
            for t in o["trace"]:
                if t["action"] in AD.LLM_ACTS:
                    n_c += 1
                elif t["action"] in AD.OPT:
                    n_p += 1
                    if t["cand"] == "c0" and t["action"] == AD.VP:
                        vp0 = True
                    if t["cand"] == "c0" and t["action"] == AD.EX:
                        ex0 = True
                    if t["cand"] != "c0" and t["action"] == AD.EX and n_c >= 2:
                        ex_new = True
            # bundle policies acquire inside the runner without per-candidate trace detail
            if pol in ("fixed_bundle", "optimizer_gated", "rule_selective"):
                n_p = o.get("n_vp", 0) + o.get("n_ex", 0)
                vp0 = o.get("n_vp", 0) > 0; ex0 = o.get("n_ex", 0) > 0; ex_new = o.get("n_ex", 0) > 1
            c["vp_c0|literals"][0] += vp0 and lit; c["vp_c0|literals"][1] += lit
            c["vp_c0|no_literals"][0] += vp0 and not lit; c["vp_c0|no_literals"][1] += not lit
            c["ex_c0"][0] += ex0; c["ex_c0"][1] += 1
            c["ex_new|>=2 cands"][0] += ex_new; c["ex_new|>=2 cands"][1] += o.get("n_llm", 1) >= 2
            probes.append(n_p)
        ch[pol] = {k: (v[0] / v[1] if v[1] else None) for k, v in c.items()}
        ch[pol]["probes_per_q"] = float(np.mean(probes)) if probes else 0.0
    out["choices"] = ch
    (d / f"policy_analysis_p{a.price}_B{a.budget}.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
