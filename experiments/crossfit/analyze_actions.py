"""Action-level analysis of the SQLite action table (results/crossfit/action_table_sqlite.jsonl).

  ORACLE : cheapest action after the first candidate that yields a correct answer
  EXP A  : marginal decision value of each evidence action (dAUROC over P1), coverage, latency
  EXP C  : DB action vs another LLM call -- dP(correct) per ms / per cost, by state region
  EXP B  : No-Probes vs All-Probes vs Adaptive-Probes (gate predicts whether a probe changes the decision)

Usage: python -m experiments.crossfit.analyze_actions
"""
from __future__ import annotations
import json
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plangate.features import p0_features, sql_features, db_features, normalize_sql
from plangate.paths import CROSSFIT_DIR
from plangate.planning.crossfit import PROBES, fp, load_action_table, triggers

rng = np.random.default_rng(0)
PREFIX = {"PARSE_BIND": "pb_", "EXPLAIN": "ex_", "VALUE_PROBE": "vp_", "EXEC_BOUNDED": "eb_"}
HAS = {"PARSE_BIND": "has_pb", "EXPLAIN": "has_ex", "VALUE_PROBE": "has_vp", "EXEC_BOUNDED": "has_eb"}


def res_agree(a, b):
    return int(fp(a) is not None and fp(a) == fp(b))


def db_ms(e, ops=PROBES):
    return sum(e["db_ms"][op] for op in ops)


# ------------------------------------------------------------------ features
def feats_base(r, items):
    f = dict(p0_features(items[r["question_id"]]))
    f.update(sql_features(r["c0"]["sql"]))
    return f


def feats_probes(r, probes):
    ev = {op: r["c0"]["ev"][op] for op in probes}
    full = db_features(ev)
    keep = {}
    for op in PROBES:
        for k, v in full.items():
            if k.startswith(PREFIX[op]) or k == HAS[op]:
                keep[k] = v if op in probes else 0
    return keep


def feats_extra(r, kind):
    c0 = r["c0"]
    if kind == "RESAMPLE_SQL":       # extra LLM call, no DB: SQL-text agreement
        return {"x_sql_agree": int(normalize_sql(c0["sql"]) == normalize_sql(r["RESAMPLE_1"]["sql"]))}
    if kind == "RESAMPLE_EXEC":      # extra LLM call + execution of both: result agreement
        return {"x_res_agree": res_agree(c0, r["RESAMPLE_1"])}
    if kind == "OTHER_MODEL_EXEC":
        return {"x_res_agree_pro": res_agree(c0, r["SWITCH_PRO"])}
    if kind == "REPAIR_CONFIRMS":    # evidence-conditioned repair leaves the SQL unchanged
        return {"x_repair_same": int(normalize_sql(c0["sql"]) == normalize_sql(r["REPAIR_DB"]["sql"]))}
    raise ValueError(kind)


def oof(X, y, repeats=10):
    """Out-of-fold probabilities, averaged over repeated stratified 5-fold CV."""
    P = np.zeros(len(y))
    for rep in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            m.fit(X[tr], y[tr])
            P[te] += m.predict_proba(X[te])[:, 1]
    return P / repeats


def mat(dicts):
    names = sorted(set().union(*dicts))
    return np.array([[d.get(n, 0) for n in names] for d in dicts], dtype=float)


def boot_delta_auc(y, p_new, p_base, n=2000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i]))
    return np.percentile(d, [2.5, 97.5])


# ------------------------------------------------------------------ analyses
def oracle_actions(rows):
    """Cheapest successful action after c0, actions ordered by mean added cost (then latency)."""
    def mean_cost(a):
        vals = [r[a]["cost"] for r in rows if r.get(a)]
        return np.mean(vals) if vals else 0.0
    order = [("ACCEPT", 0.0), ("VALUE_FIX (DB only)", 0.0)] + sorted(
        [(a, mean_cost(a)) for a in ["REPAIR_DB", "REPAIR_BLIND", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"]],
        key=lambda x: x[1])
    cats = Counter()
    for r in rows:
        if r["c0"]["correct"]:
            cats["ACCEPT"] += 1; continue
        if r["VALUE_FIX"] and r["VALUE_FIX"]["correct"]:
            cats["VALUE_FIX (DB only)"] += 1; continue
        for a, _ in order[2:]:
            if r.get(a) and r[a]["correct"]:
                cats[a] += 1; break
        else:
            pool = [r[a]["correct"] for a in ("RESAMPLE_2",) if r.get(a)]
            cats["needs >=2 extra candidates" if any(pool) else "none succeeds"] += 1
    n = len(rows)
    print("\n== ORACLE: cheapest successful action after the first candidate (flash_direct)")
    print("   (actions tried in order of mean added cost: " + ", ".join(f"{a}={c:.2f}" for a, c in order) + ")")
    for a, _ in order + [("needs >=2 extra candidates", 0), ("none succeeds", 0)]:
        print(f"  {cats[a]/n:6.1%}  {a}")

    wrong = [r for r in rows if not r["c0"]["correct"]]
    print(f"\n  Non-exclusive rescue rate among the {len(wrong)} wrong first candidates:")
    rescue = {}
    for a in ["VALUE_FIX", "REPAIR_DB", "REPAIR_BLIND", "RESAMPLE_1", "RESAMPLE_2", "SWITCH_PRO", "SWITCH_QWEN"]:
        k = sum(1 for r in wrong if r.get(a) and r[a]["correct"])
        rescue[a] = k / len(wrong)
        print(f"    {a:13s} {k:3d}  ({k/len(wrong):.1%})")
    # does the rescuing action depend on the state? split by DB-flag
    print("\n  Rescue rate by state region (wrong first candidates):")
    for name, cond in (("DB-flagged", lambda r: triggers(r["c0"]["ev"])),
                       ("not flagged", lambda r: not triggers(r["c0"]["ev"]))):
        sub = [r for r in wrong if cond(r)]
        print(f"    {name:12s} n={len(sub):3d}  " + "  ".join(
            f"{a}={np.mean([bool(r.get(a) and r[a]['correct']) for r in sub]):.2f}"
            for a in ["VALUE_FIX", "REPAIR_DB", "REPAIR_BLIND", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"]))
    return {"distribution": {a: cats[a] / n for a in cats}, "rescue_rate_wrong": rescue}


def exp_a(rows, items):
    y = np.array([int(r["c0"]["correct"]) for r in rows])
    base_d = [feats_base(r, items) for r in rows]
    Pb = oof(mat(base_d), y)
    auc_base = roc_auc_score(y, Pb)
    print("\n== EXP A: marginal decision value of each action  (target: is first candidate correct?)")
    print(f"   base P1 (question+schema+candidate SQL) AUROC = {auc_base:.3f}; LogisticRegression, 10x5-fold OOF")
    print(f"  {'action':22s} {'dAUROC':>7} {'95% CI':>17} {'applic.':>8} {'fires':>6} {'med ms':>8} {'p95 ms':>8} {'cost':>6}")
    table = {}

    def report(name, extra, applic, fires, ms, cost):
        d = [{**b, **e} for b, e in zip(base_d, extra)]
        P = oof(mat(d), y)
        lo, hi = boot_delta_auc(y, P, Pb)
        dA = roc_auc_score(y, P) - auc_base
        table[name] = {"dAUROC": dA, "ci": [lo, hi], "applicable": applic, "fires": fires,
                       "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
                       "cost": float(np.mean(cost))}
        print(f"  {name:22s} {dA:+7.3f} [{lo:+.3f},{hi:+.3f}] {applic:8.1%} {fires:6.1%} "
              f"{np.median(ms):8.1f} {np.percentile(ms, 95):8.1f} {np.mean(cost):6.2f}")
        return P

    for op in PROBES:
        ms = [r["c0"]["db_ms"][op] for r in rows]
        ev = [r["c0"]["ev"][op] for r in rows]
        if op == "PARSE_BIND":
            applic, fires = 1.0, np.mean([not e.get("valid") for e in ev])
        elif op == "EXPLAIN":
            applic, fires = np.mean([e.get("valid", False) for e in ev]), np.mean([bool(e.get("suspicious_plan")) for e in ev])
        elif op == "VALUE_PROBE":
            applic, fires = np.mean([e.get("supported", False) for e in ev]), np.mean([bool(e.get("n_missing")) for e in ev])
        else:
            applic = 1.0
            fires = np.mean([(not e.get("ok")) or e.get("empty") or e.get("all_null") for e in ev])
        report(op, [feats_probes(r, [op]) for r in rows], applic, fires, ms, [m / 1000 for m in ms])
    ms_all = [db_ms(r["c0"]) for r in rows]
    P_all = report("ALL_DB_PROBES", [feats_probes(r, PROBES) for r in rows], 1.0,
                   np.mean([triggers(r["c0"]["ev"]) for r in rows]), ms_all, [m / 1000 for m in ms_all])
    for kind, act, needs_exec in (("RESAMPLE_SQL", "RESAMPLE_1", False), ("RESAMPLE_EXEC", "RESAMPLE_1", True),
                                  ("OTHER_MODEL_EXEC", "SWITCH_PRO", True), ("REPAIR_CONFIRMS", "REPAIR_DB", False)):
        ex = [feats_extra(r, kind) for r in rows]
        key = list(ex[0])[0]
        fires = np.mean([1 - e[key] for e in ex])   # disagreement rate
        ms = [r[act]["llm_ms"] + (r["c0"]["db_ms"]["EXEC_BOUNDED"] + r[act]["db_ms"]["EXEC_BOUNDED"] if needs_exec else 0)
              + (db_ms(r["c0"]) if act == "REPAIR_DB" else 0) for r in rows]
        cost = [r[act]["cost"] for r in rows]
        report(f"+LLM {kind}", ex, 1.0, fires, ms, cost)
    # all DB probes + execution agreement with a resample: the strongest state
    ex = [{**feats_probes(r, PROBES), **feats_extra(r, "RESAMPLE_EXEC")} for r in rows]
    ms = [db_ms(r["c0"]) + r["RESAMPLE_1"]["llm_ms"] + r["RESAMPLE_1"]["db_ms"]["EXEC_BOUNDED"] for r in rows]
    report("ALL_DB + RESAMPLE_EXEC", ex, 1.0, 0.0, ms, [r["RESAMPLE_1"]["cost"] for r in rows])
    return table, y, Pb, P_all


def exp_c(rows, P_all):
    """dP(correct) per action, relative to keeping c0, by region of the post-probe state."""
    def final(r, a):
        if a == "KEEP":
            return r["c0"]["correct"]
        if a == "VALUE_FIX":
            return r["VALUE_FIX"]["correct"] if r["VALUE_FIX"] else r["c0"]["correct"]
        return r[a]["correct"]

    def added(r, a):
        if a == "KEEP":
            return 0.0, 0.0
        if a == "VALUE_FIX":   # needs the value probe (+ re-execution if rewritten)
            ms = r["c0"]["db_ms"]["VALUE_PROBE"] + (r["VALUE_FIX"]["db_ms"]["EXEC_BOUNDED"] if r["VALUE_FIX"] else 0)
            return ms, ms / 1000
        ms = r[a]["llm_ms"] + (db_ms(r["c0"]) if a == "REPAIR_DB" else 0)
        return ms, r[a]["cost"] + (db_ms(r["c0"]) / 1000 if a == "REPAIR_DB" else 0)

    regions = {
        "all": lambda i, r: True,
        "DB-flagged": lambda i, r: triggers(r["c0"]["ev"]),
        "unflagged & uncertain (p<0.6)": lambda i, r: not triggers(r["c0"]["ev"]) and P_all[i] < 0.6,
        "unflagged & confident (p>=0.6)": lambda i, r: not triggers(r["c0"]["ev"]) and P_all[i] >= 0.6,
    }
    acts = ["VALUE_FIX", "REPAIR_DB", "REPAIR_BLIND", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"]
    print("\n== EXP C: DB action vs another LLM call  (replace c0 by the action's output)")
    print("   dP = acc(action) - acc(keep c0);  per-second and per-cost-unit efficiency of the added work")
    out = {}
    for rname, cond in regions.items():
        sub = [r for i, r in enumerate(rows) if cond(i, r)]
        if not sub:
            continue
        base = np.mean([final(r, "KEEP") for r in sub])
        print(f"\n  region: {rname}  (n={len(sub)}, acc(keep)={base:.3f})")
        print(f"    {'action':13s} {'acc':>6} {'dP':>7} {'+ms':>8} {'+cost':>6} {'dP/s':>8} {'dP/cost':>8}")
        out[rname] = {}
        for a in acts:
            acc = np.mean([final(r, a) for r in sub])
            ms = np.mean([added(r, a)[0] for r in sub]); cost = np.mean([added(r, a)[1] for r in sub])
            dp = acc - base
            out[rname][a] = {"acc": acc, "dP": dp, "ms": ms, "cost": cost}
            print(f"    {a:13s} {acc:6.3f} {dp:+7.3f} {ms:8.0f} {cost:6.2f} {dp/max(ms/1000,1e-6):+8.3f} "
                  f"{dp/max(cost,1e-6):+8.3f}")
    return out


def exp_b(rows, items, y, fallback="REPAIR_DB", taus=(0.4, 0.5, 0.6, 0.7)):
    """Two-stage policy: c0 -> probes -> ACCEPT if P(correct) >= tau else fallback action.

    Acceptor is trained on all 16 probe subsets (missing probes = has_* flags 0), so it can
    decide from any partial evidence. The adaptive gate runs probes cheap->expensive and runs
    a probe only if it predicts the probe would flip the accept/fallback decision.
    """
    n = len(rows)
    subsets = [list(s) for k in range(len(PROBES) + 1) for s in combinations(PROBES, k)]
    base_d = [feats_base(r, items) for r in rows]
    names = sorted(set().union(*[{**base_d[0], **feats_probes(rows[0], PROBES)}]))
    vec = lambda i, S: [({**base_d[i], **feats_probes(rows[i], S)}).get(k, 0) for k in names]

    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(folds):
        fold_of[te] = f
    acceptors = []
    for tr, _ in folds:
        X = np.array([vec(i, S) for i in tr for S in subsets]); yy = np.array([y[i] for i in tr for S in subsets])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X, yy)
        acceptors.append(m)
    P = {}
    for i in range(n):
        m = acceptors[fold_of[i]]
        ps = m.predict_proba(np.array([vec(i, S) for S in subsets]))[:, 1]
        for S, p in zip(subsets, ps):
            P[(i, tuple(S))] = p

    def outcome(i, accept):
        r = rows[i]
        return r["c0"]["correct"] if accept else r[fallback]["correct"]

    needs_ev = fallback == "REPAIR_DB"

    def work(i, S, accept):
        """(DB ms, LLM cost) actually spent: probes run + fallback (+ the probes repair needs)."""
        ms = db_ms(rows[i]["c0"], S)
        cost = rows[i]["c0"]["cost"]
        if not accept:
            cost += rows[i][fallback]["cost"]
            if needs_ev:
                ms += db_ms(rows[i]["c0"], [o for o in PROBES if o not in S])
        return ms, cost

    results = {}
    print(f"\n== EXP B: do I need this probe?  (c0 -> probes -> ACCEPT or {fallback})")
    print(f"  {'policy':28s} {'tau':>4} {'acc':>6} {'DB ms/q':>8} {'p95 DBms':>9} {'probes/q':>8} "
          f"{'fallback%':>9} {'LLM cost':>8}")
    for tau in taus:
        for pol in ("No-Probes", "All-Probes"):
            S = () if pol == "No-Probes" else tuple(PROBES)
            dec = [P[(i, S)] >= tau for i in range(n)]
            acc = np.mean([outcome(i, d) for i, d in enumerate(dec)])
            w = [work(i, S, d) for i, d in enumerate(dec)]
            ms = [x[0] for x in w]
            print(f"  {pol:28s} {tau:4.2f} {acc:6.3f} {np.mean(ms):8.1f} {np.percentile(ms,95):9.1f} {len(S):8.2f} "
                  f"{1-np.mean(dec):9.1%} {np.mean([x[1] for x in w]):8.2f}")
            results[(pol, tau)] = (acc, float(np.mean(ms)), float(np.mean([x[1] for x in w])))
        # gate: label = does adding probe op to current set S flip the decision?  (cross-fitted)
        G = {}
        for op in PROBES:
            Xg, yg, idx = [], [], []
            for i in range(n):
                for S in subsets:
                    if op in S or any(PROBES.index(o) > PROBES.index(op) for o in S):
                        continue   # greedy order: only states reachable before op
                    S2 = tuple(sorted(set(S) | {op}, key=PROBES.index))
                    flip = (P[(i, tuple(S))] >= tau) != (P[(i, S2)] >= tau)
                    Xg.append(vec(i, list(S)) + [P[(i, tuple(S))]]); yg.append(int(flip)); idx.append((i, tuple(S)))
            Xg = np.array(Xg); yg = np.array(yg)
            for f, (tr, te) in enumerate(folds):
                trm = np.array([fold_of[i] != f for i, _ in idx]); tem = ~trm
                if yg[trm].sum() < 3:
                    for k in np.where(tem)[0]:
                        G[(op,) + idx[k]] = 0.0
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
                m.fit(Xg[trm], yg[trm])
                for k, p in zip(np.where(tem)[0], m.predict_proba(Xg[tem])[:, 1]):
                    G[(op,) + idx[k]] = p
        for gamma in (0.3, 0.5, 0.7, 0.85):
            accs, mss, nps, fbs, costs = [], [], [], [], []
            for i in range(n):
                S = []
                for op in PROBES:
                    if G.get((op, i, tuple(S)), 0.0) >= gamma:
                        S.append(op)
                p = P[(i, tuple(S))]
                ms, cost = work(i, S, p >= tau)
                accs.append(outcome(i, p >= tau)); mss.append(ms); nps.append(len(S)); fbs.append(p < tau)
                costs.append(cost)
            print(f"  {'Adaptive (gate>=' + str(gamma) + ')':28s} {tau:4.2f} {np.mean(accs):6.3f} {np.mean(mss):8.1f} "
                  f"{np.percentile(mss,95):9.1f} {np.mean(nps):8.2f} {np.mean(fbs):9.1%} {np.mean(costs):8.2f}")
            results[(f"Adaptive{gamma}", tau)] = (float(np.mean(accs)), float(np.mean(mss)), float(np.mean(costs)))
    return results


def main():
    rows, items = load_action_table("sqlite")
    print(f"{len(rows)} questions; first candidate = flash_direct, acc={np.mean([r['c0']['correct'] for r in rows]):.3f}")
    rep = {"oracle": oracle_actions(rows)}
    table, y, Pb, P_all = exp_a(rows, items)
    rep["exp_a"] = table
    rep["exp_c"] = exp_c(rows, P_all)
    rep["exp_b"] = {}
    for fb in ("REPAIR_DB", "SWITCH_QWEN"):
        rep["exp_b"][fb] = {f"{k[0]}@{k[1]}": v for k, v in exp_b(rows, items, y, fallback=fb).items()}
    (CROSSFIT_DIR / "action_report_sqlite.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf8")


if __name__ == "__main__":
    main()
