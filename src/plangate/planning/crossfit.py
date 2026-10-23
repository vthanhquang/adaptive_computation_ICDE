"""Cross-fitted sequential planner over the action tables of the BIRD dev samples.

State    s = set S of acquired candidates (+ their DB evidence, depending on the variant)
Actions  ACCEPT | VALUE_FIX (DB only) | RESAMPLE (flash, T=0.8; R1 then R2) | REPAIR_DB | SWITCH_PRO | SWITCH_QWEN
Answer   argmax_g A(g | S): verified replacement, a new candidate wins only if the acceptor prefers it
Policy   a* = argmax_a Q(s,a) - lam*cost(a); act iff Q(s,a*) - V(s) > lam*cost(a*); at most 3 actions

Variants (the DBMS ablation):
  full      : all 4 DB probes on every candidate, execution-agreement clusters, VALUE_FIX, REPAIR_DB
  exec_only : only bounded execution (agreement + result shape); no value probe / VALUE_FIX / REPAIR_DB
  nodb      : no DB access at all; SQL-text agreement only; REPAIR_BLIND instead of REPAIR_DB

All learned parts are cross-fitted with GroupKFold over questions.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import benchmark
from ..features import p0_features, sql_features, db_features, normalize_sql
from ..jsonl import read_jsonl
from ..paths import CROSSFIT_DIR

PROBES = ["PARSE_BIND", "VALUE_PROBE", "EXPLAIN", "EXEC_BOUNDED"]   # ordered by typical cost
SRC = {"c0": "flash", "RESAMPLE_1": "resample", "RESAMPLE_2": "resample", "REPAIR_DB": "repair",
       "REPAIR_BLIND": "repair", "VALUE_FIX": "valuefix", "SWITCH_PRO": "pro", "SWITCH_QWEN": "qwen"}
VARIANT_ACTIONS = {
    "full": ["VALUE_FIX", "RESAMPLE_1", "RESAMPLE_2", "REPAIR_DB", "SWITCH_PRO", "SWITCH_QWEN"],
    "exec_only": ["RESAMPLE_1", "RESAMPLE_2", "REPAIR_BLIND", "SWITCH_PRO", "SWITCH_QWEN"],
    "nodb": ["RESAMPLE_1", "RESAMPLE_2", "REPAIR_BLIND", "SWITCH_PRO", "SWITCH_QWEN"],
}
MAX_ACTIONS = 3


ENGINES = ("sqlite", "pg")


def load_action_table(engine):
    """Action-table rows of an engine sorted by question id, and the benchmark items by question id.
    sqlite: the 300-question BIRD dev sample; pg: the 500 BIRD mini-dev entries (498 unique questions)."""
    rows = sorted(read_jsonl(CROSSFIT_DIR / f"action_table_{engine}.jsonl"), key=lambda r: r["question_id"])
    source = benchmark.load_dev() if engine == "sqlite" else benchmark.load_minidev()
    return rows, {x["question_id"]: x for x in source}


def triggers(ev):
    """Rule-based DB flag on a candidate: bind error, join without condition, missing literal,
    execution failure, or an empty / all-NULL result."""
    pb, ex, vp, eb = (ev[k] for k in ("PARSE_BIND", "EXPLAIN", "VALUE_PROBE", "EXEC_BOUNDED"))
    return bool((not pb.get("valid")) or (ex.get("valid") and ex.get("joins_without_on")) or vp.get("n_missing")
                or (pb.get("valid") and not eb.get("ok")) or (eb.get("ok") and (eb.get("empty") or eb.get("all_null"))))


def question_folds(n, k=5):
    """GroupKFold over question indices: (fold index of every question, training indices per fold)."""
    folds_idx = list(GroupKFold(k).split(np.zeros(n), groups=np.arange(n)))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(folds_idx):
        fold_of[te] = f
    return fold_of, [tr for tr, _ in folds_idx]


def fp(e):
    """Result fingerprint of an executed candidate, or None (failed or not executed)."""
    eb = e["ev"].get("EXEC_BOUNDED")
    return eb.get("fingerprint") if eb and eb.get("ok") else None


def probes_for(variant):
    return {"full": PROBES, "exec_only": ["EXEC_BOUNDED"], "nodb": []}[variant]


def cand_cost(r, name, variant):
    """(cost units, latency ms) of acquiring candidate `name` incl. its DB probes."""
    e = r[name]
    db = sum(e["db_ms"][op] for op in probes_for(variant))
    if name == "VALUE_FIX":   # DB-only: probe on c0 was already paid; re-probe the rewritten SQL
        return db / 1000, db
    if name == "REPAIR_DB" and variant == "full":
        return e["cost"] + db / 1000, e["llm_ms"] + db
    return e["cost"] + db / 1000, e["llm_ms"] + db


def available(r, name):
    return r.get(name) is not None


def valid_set(T):
    return not ("RESAMPLE_2" in T and "RESAMPLE_1" not in T)


# ------------------------------------------------------------------ features
def acc_feats(r, item, S, g, variant):
    e = r[g]
    f = dict(p0_features(item))
    f.update(sql_features(e["sql"]))
    for s in set(SRC.values()):
        f["src_" + s] = int(SRC[g] == s)
    f["n_cands"] = len(S)
    others = [x for x in S if x != g]
    ns = normalize_sql(e["sql"])
    f["sql_agree_n"] = sum(normalize_sql(r[o]["sql"]) == ns for o in others)
    if variant == "nodb":
        return f
    ev = e["ev"] if variant == "full" else {"EXEC_BOUNDED": e["ev"]["EXEC_BOUNDED"]}
    f.update(db_features(ev))
    fps = [fp(r[x]) for x in S]
    mine = fp(e)
    clusters = Counter(x for x in fps if x is not None)
    f["res_agree_n"] = (clusters[mine] - 1) if mine else 0
    f["res_agree_frac"] = f["res_agree_n"] / max(len(others), 1)
    f["is_plurality"] = int(bool(mine) and clusters[mine] == max(clusters.values()))
    f["largest_cluster"] = max(clusters.values()) / len(S) if clusters else 0
    f["n_clusters"] = len(clusters)
    f["exec_fail"] = int(mine is None)
    return f


def state_feats(r, item, S, best, a_best, a_second, variant, trigger_fn=triggers):
    f = {"st_" + k: v for k, v in acc_feats(r, item, S, best, variant).items()}
    f["A_best"] = a_best; f["A_second"] = a_second; f["A_margin"] = a_best - a_second
    for a in set(VARIANT_ACTIONS["full"]) | {"REPAIR_BLIND"}:
        f["done_" + a] = int(a in S)
    if variant == "full":
        f["c0_flag"] = int(trigger_fn(r["c0"]["ev"]))
        f["valuefix_avail"] = int(available(r, "VALUE_FIX"))
    return f


class CF:
    """Cross-fitted classifier keyed by question group: model i is fitted on the questions of
    folds[i]. A fold without rows or with a single class degrades to a constant prediction."""

    def __init__(self, kind="gb"):
        self.kind = kind

    def fit(self, X, y, groups, folds):
        self.names = sorted(set().union(*X)) if X else []
        M = self._m(X); y = np.array(y); g = np.array(groups)
        self.models = []
        for tr_q in folds:
            idx = np.isin(g, tr_q)
            if idx.sum() == 0 or len(set(y[idx].tolist())) < 2:
                self.models.append(float(y[idx].mean()) if idx.sum() else 0.5)
                continue
            if self.kind == "gb":
                m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                   l2_regularization=1.0, random_state=0)
            else:
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
            m.fit(M[idx], y[idx]); self.models.append(m)
        return self

    def _m(self, X):
        return np.array([[d.get(n, 0) for n in self.names] for d in X], dtype=float)

    def predict(self, X, fold_ids):
        M = self._m(X); out = np.zeros(len(X)); fold_ids = np.array(fold_ids)
        for i, m in enumerate(self.models):
            idx = fold_ids == i
            if idx.any():
                out[idx] = m if isinstance(m, float) else m.predict_proba(M[idx])[:, 1]
        return out


# ------------------------------------------------------------------ planner training
def build_planner(rows, items, variant, fold_of, train_folds, selector="pairwise", trigger_fn=triggers):
    """Fits acceptor A, pairwise selector P, state value V and action value Q (cross-fitted) and
    returns their out-of-fold values on every reachable state. `trigger_fn` is the rule-based DB flag
    used as a state feature (overridable for tables with partially acquired evidence)."""
    acts = VARIANT_ACTIONS[variant]
    # 1) acceptor over all candidate sets {c0} U T, |T| <= MAX_ACTIONS
    AX, Ay, Ag, Akeys = [], [], [], []
    sets = {}
    for i, r in enumerate(rows):
        avail = [a for a in acts if available(r, a)]
        ss = []
        for k in range(MAX_ACTIONS + 1):
            for T in combinations(avail, k):
                if valid_set(T):
                    ss.append(("c0",) + T)
        sets[i] = ss
        for S in ss:
            for g in S:
                AX.append(acc_feats(r, items[r["question_id"]], S, g, variant))
                Ay.append(int(r[g]["correct"])); Ag.append(i); Akeys.append((i, S, g))
    A = CF().fit(AX, Ay, Ag, train_folds)
    Ap = A.predict(AX, [fold_of[i] for i in Ag])
    Aval = {k: p for k, p in zip(Akeys, Ap)}

    def key_of(r, g):
        """Candidates with identical evidence-level outputs form one cluster."""
        if variant == "nodb":
            return "sql:" + normalize_sql(r[g]["sql"])
        f = fp(r[g])
        return ("fp:" + f) if f else ("fail:" + g)

    def pointwise(i, S):
        ps = sorted(((Aval[(i, S, g)], g) for g in S), reverse=True)
        return ps[0][1], ps[0][0], (ps[1][0] if len(ps) > 1 else 0.0)

    if selector == "pointwise":
        select = pointwise
    else:
        # pairwise preference between cluster representatives, cross-fitted
        def reps(i, S):
            r = rows[i]; cl = defaultdict(list)
            for g in S:
                cl[key_of(r, g)].append(g)
            return [max(gs, key=lambda g: Aval[(i, S, g)]) for gs in cl.values()], cl

        def pfeat(i, S, a, b):
            r, it = rows[i], items[rows[i]["question_id"]]
            fa = acc_feats(r, it, S, a, variant); fb = acc_feats(r, it, S, b, variant)
            f = {"d_" + k: fa.get(k, 0) - fb.get(k, 0) for k in set(fa) | set(fb)}
            for s in set(SRC.values()):
                f["a_" + s] = int(SRC[a] == s); f["b_" + s] = int(SRC[b] == s)
            f["Aa"] = Aval[(i, S, a)]; f["Ab"] = Aval[(i, S, b)]
            return f

        PX, Py, Pg, Pkeys = [], [], [], []
        for i, r in enumerate(rows):
            for S in sets[i]:
                rp, _ = reps(i, S)
                for a in rp:
                    for b in rp:
                        if a == b:
                            continue
                        PX.append(pfeat(i, S, a, b)); Pkeys.append((i, S, a, b)); Pg.append(i)
                        Py.append(int(r[a]["correct"]))
        # train only on informative pairs (exactly one correct); predict for all
        Pm = np.array([rows[k[0]][k[2]]["correct"] != rows[k[0]][k[3]]["correct"] for k in Pkeys])
        Pmodel = CF().fit([x for x, m in zip(PX, Pm) if m], [y for y, m in zip(Py, Pm) if m],
                          [g for g, m in zip(Pg, Pm) if m], train_folds)
        Pval = dict(zip(Pkeys, Pmodel.predict(PX, [fold_of[i] for i in Pg])))

        def select(i, S):
            rp, _ = reps(i, S)
            if len(rp) == 1:
                g = rp[0]
            else:
                score = {a: sum(Pval[(i, S, a, b)] - Pval[(i, S, b, a)] for b in rp if b != a) for a in rp}
                g = max(rp, key=lambda a: (score[a], Aval[(i, S, a)]))
            others = [Aval[(i, S, x)] for x in S if x != g]
            return g, Aval[(i, S, g)], (max(others) if others else 0.0)

    # 2) V(s) and Q(s,a) on states with |T| < MAX_ACTIONS
    VX, Vy, Vg, Vkeys, QX, Qy, Qg, Qkeys = [], [], [], [], [], [], [], []
    canon = {}
    for i, r in enumerate(rows):
        for S in sets[i]:
            canon[(i, frozenset(S))] = S
    for i, r in enumerate(rows):
        it = items[r["question_id"]]
        for S in sets[i]:
            best, ab, a2 = select(i, S)
            sf = state_feats(r, it, S, best, ab, a2, variant, trigger_fn)
            VX.append(sf); Vy.append(int(r[best]["correct"])); Vg.append(i); Vkeys.append((i, S))
            if len(S) - 1 >= MAX_ACTIONS:
                continue
            for a in acts:
                S2 = canon.get((i, frozenset(S + (a,))))
                if a in S or S2 is None:
                    continue
                b2, _, _ = select(i, S2)
                qf = dict(sf)
                for x in acts:
                    qf["act_" + x] = int(x == a)
                QX.append(qf); Qy.append(int(r[b2]["correct"])); Qg.append(i); Qkeys.append((i, S, a))
    V = CF().fit(VX, Vy, Vg, train_folds)
    Q = CF().fit(QX, Qy, Qg, train_folds)
    Vval = dict(zip(Vkeys, V.predict(VX, [fold_of[i] for i in Vg])))
    Qval = dict(zip(Qkeys, Q.predict(QX, [fold_of[i] for i in Qg])))
    return {"select": select, "V": Vval, "Q": Qval, "canon": canon, "acts": acts,
            "A": Aval, "P": (Pval if selector != "pointwise" else {})}


def run_planner(rows, P, variant, lam, lat_w=0.0):
    """lam: penalty per cost unit; lat_w: extra penalty per second of latency."""
    out = []
    for i, r in enumerate(rows):
        S = ("c0",)
        cost, lat = cand_cost(r, "c0", variant)
        path = []
        while len(S) - 1 < MAX_ACTIONS:
            v = P["V"][(i, S)]
            best_a, best_score = None, 0.0
            for a in P["acts"]:
                if a in S or (i, S, a) not in P["Q"]:
                    continue
                c, l = cand_cost(r, a, variant)
                score = P["Q"][(i, S, a)] - v - lam * c - lat_w * l / 1000
                if score > best_score:
                    best_a, best_score = a, score
            if best_a is None:
                break
            c, l = cand_cost(r, best_a, variant)
            cost += c; lat += l; path.append(best_a)
            S = P["canon"][(i, frozenset(S + (best_a,)))]
        g, _, _ = P["select"](i, S)
        out.append({"correct": int(r[g]["correct"]), "cost": cost, "lat": lat, "plan": "->".join(path) or "ACCEPT",
                    "final": g, "n_llm": 1 + sum(a != "VALUE_FIX" for a in path)})
    return out


def run_hybrid(rows, P, variant, lam, tau, strong="SWITCH_QWEN", max_cheap=2):
    """Two-phase policy: (1) cheap DB-guided local actions chosen by Q/V (no strong model),
    (2) escalate to the strong model iff V(s) < tau; final answer = DB-verified selection."""
    out = []
    cheap = [a for a in P["acts"] if a != strong]
    for i, r in enumerate(rows):
        S = ("c0",)
        cost, lat = cand_cost(r, "c0", variant)
        path = []
        while len(S) - 1 < max_cheap:
            v = P["V"][(i, S)]
            best_a, best_score = None, 0.0
            for a in cheap:
                if a in S or (i, S, a) not in P["Q"]:
                    continue
                c, l = cand_cost(r, a, variant)
                score = P["Q"][(i, S, a)] - v - lam * c
                if score > best_score:
                    best_a, best_score = a, score
            if best_a is None:
                break
            c, l = cand_cost(r, best_a, variant)
            cost += c; lat += l; path.append(best_a)
            S = P["canon"][(i, frozenset(S + (best_a,)))]
        if P["V"][(i, S)] < tau and available(r, strong):
            S2 = P["canon"].get((i, frozenset(S + (strong,))))
            if S2 is not None:
                c, l = cand_cost(r, strong, variant)
                cost += c; lat += l; path.append(strong); S = S2
        g, _, _ = P["select"](i, S)
        out.append({"correct": int(r[g]["correct"]), "cost": cost, "lat": lat, "plan": "->".join(path) or "ACCEPT",
                    "final": g, "n_llm": 1 + sum(a != "VALUE_FIX" for a in path)})
    return out


# ------------------------------------------------------------------ baselines
def vote(r, names):
    """Execution-result majority vote; ties broken by list order. Failed executions never win."""
    fps = [fp(r[n]) for n in names]
    cnt = Counter(x for x in fps if x)
    if not cnt:
        return names[0]
    top = max(cnt.values())
    for n, x in zip(names, fps):
        if x and cnt[x] == top:
            return n


def res(r, final, names, variant="full", parallel=False):
    costs = [cand_cost(r, n, variant) for n in names]
    lat = max(l for _, l in costs) if parallel else sum(l for _, l in costs)
    return {"correct": int(r[final]["correct"]), "cost": sum(c for c, _ in costs), "lat": lat,
            "plan": "+".join(names), "final": final, "n_llm": len(names)}


def baselines(rows, items, fold_of, train_folds, M, published=None):
    """Non-PlanGate policies on the same action table. `M`: generation matrix rows by question
    (extra fixed models, SQLite only); `published`: per-question outputs of re-implemented methods."""
    B = defaultdict(list)   # name -> list of (label, per-question results)
    for name, src in (("fixed flash", "c0"), ("fixed pro", "SWITCH_PRO"), ("fixed qwen", "SWITCH_QWEN")):
        B[name].append(("", [res(r, src, [src], "nodb") for r in rows]))
    for cfg in ("glm_direct", "pro_decompose", "kimi_decompose") if M else ():
        B["fixed " + cfg].append(("", [{"correct": int(M[r["question_id"]][cfg]["correct"]),
                                        "cost": M[r["question_id"]][cfg]["cost"],
                                        "lat": M[r["question_id"]][cfg]["latency_ms"], "plan": cfg,
                                        "n_llm": 1} for r in rows]))
    # one-shot router over {flash, pro, qwen}: P0 features, cross-fitted per-config logistic models
    names = ["c0", "SWITCH_PRO", "SWITCH_QWEN"]
    X = [p0_features(items[r["question_id"]]) for r in rows]
    probs = {}
    for n in names:
        y = [int(r[n]["correct"]) for r in rows]
        m = CF("lr").fit(X, y, list(range(len(rows))), train_folds)
        probs[n] = m.predict(X, [fold_of[i] for i in range(len(rows))])
    mc = {n: np.mean([r[n]["cost"] for r in rows]) for n in names}
    for lam in np.linspace(0, 0.1, 21):
        out = []
        for i, r in enumerate(rows):
            n = max(names, key=lambda n: probs[n][i] - lam * mc[n])
            out.append(res(r, n, [n], "nodb"))
        B["one-shot router"].append((f"lam={lam:.3f}", out))
    # rule cascade: flash -> any DB trigger -> qwen (replace)
    B["rule cascade (DB trigger->qwen)"].append(("", [
        res(r, "SWITCH_QWEN" if triggers(r["c0"]["ev"]) else "c0",
            ["c0", "SWITCH_QWEN"] if triggers(r["c0"]["ev"]) else ["c0"]) for r in rows]))
    # agreement cascade (MoT-style): flash + resample in parallel; agree -> accept, else qwen (replace)
    out = []
    for r in rows:
        a = fp(r["c0"]); b = fp(r["RESAMPLE_1"])
        first = res(r, "c0", ["c0", "RESAMPLE_1"], "exec_only", parallel=True)
        if a and a == b:
            out.append(first)
        else:
            q = res(r, "SWITCH_QWEN", ["SWITCH_QWEN"], "exec_only")
            out.append({**q, "cost": first["cost"] + q["cost"], "lat": first["lat"] + q["lat"], "n_llm": 3})
    B["agreement cascade (MoT-style)"].append(("", out))
    # self-consistency and PRISM-like ensemble budget (parallel, execution vote)
    for name, cands in (("self-consistency@3 (flash)", ["c0", "RESAMPLE_1", "RESAMPLE_2"]),
                        ("self-consistency@2 (flash)", ["c0", "RESAMPLE_1"]),
                        ("ensemble@5 exec-vote", ["c0", "RESAMPLE_1", "RESAMPLE_2", "SWITCH_PRO", "SWITCH_QWEN"]),
                        ("ensemble@4 exec-vote", ["c0", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"]),
                        ("ensemble@3 models exec-vote", ["c0", "SWITCH_PRO", "SWITCH_QWEN"])):
        if all(available(rows[0], c) for c in cands):
            B[name].append(("", [res(r, vote(r, cands), cands, "exec_only", parallel=True) for r in rows]))
    # published methods re-implemented in this setting (experiments/baselines/published_methods.py)
    if published:
        d = published
        pos = {q: i for i, q in enumerate(d["question_ids"])}
        for name, outs in d["policies"].items():
            # one family per method variant; sweep parameters (theta / t) become points of that family
            fam = "[paper] " + re.sub(r"\s*\(theta=[^)]*\)|, t=[^)]*", "", name)
            B[fam].append((name, [dict(outs[pos[r["question_id"]]], plan=name) for r in rows]))
    return B


def confidence_cascades(rows, P_full, lamless=None):
    """AutoMix-style: flash -> if A(c0|{c0}) < tau -> qwen. 'replace' vs 'select' (verified)."""
    out = defaultdict(list)
    for tau in np.linspace(0.1, 0.95, 18):
        rep, sel = [], []
        for i, r in enumerate(rows):
            _, p, _ = P_full["select"](i, ("c0",))
            if p >= tau:
                x = res(r, "c0", ["c0"]); rep.append(x); sel.append(x)
            else:
                S = P_full["canon"].get((i, frozenset(("c0", "SWITCH_QWEN"))))
                g, _, _ = P_full["select"](i, S)
                rep.append(res(r, "SWITCH_QWEN", ["c0", "SWITCH_QWEN"]))
                sel.append(res(r, g, ["c0", "SWITCH_QWEN"]))
        out["confidence cascade, replace (AutoMix-like)"].append((f"tau={tau:.2f}", rep))
        out["confidence cascade, verified select"].append((f"tau={tau:.2f}", sel))
    return out


def summarize(results):
    return (float(np.mean([x["correct"] for x in results])), float(np.mean([x["cost"] for x in results])),
            float(np.mean([x["lat"] for x in results])), float(np.percentile([x["lat"] for x in results], 95)),
            float(np.mean([x["n_llm"] for x in results])))


def pareto(points, xi):
    pts = sorted(points, key=lambda p: (p[xi], -p[0]))
    out, best = [], -1
    for p in pts:
        if p[0] > best + 1e-12:
            out.append(p); best = p[0]
    return out
