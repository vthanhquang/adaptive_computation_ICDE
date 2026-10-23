"""Online learning from the system's own logs.

Builds on online_learning.py. Three changes, all replayed from the cache (no LLM call):

1. Advantage model.  A planner trained on its own logs over-acts: V(s) is learned
   on every logged question, Q(s,a) only on the questions where a was taken, so Q - V is biased.
   ADV(s,a) regresses the within-question gain  correct(select(S+a)) - correct(select(S))
   directly; with epsilon exploration the remaining selection bias depends only on s.
2. DB-consistency self-labels (no gold).  A candidate is labelled wrong if the DB says so (bind error,
   execution failure, empty / all-NULL result, a literal that does not exist), labelled correct
   if it is DB-clean and its result is shared by >= 2 model families, and left unlabelled otherwise
   (unlabelled candidates are dropped from the log record; the question is dropped if c0 is).
3. Active audit.  A fraction beta of the questions of each block receives correctness feedback
   (a human / downstream check) on its acquired candidates; the audited questions are the ones
   whose final state value V(s) is closest to 0.5 (uncertainty sampling), or random ones.

Usage:  python -m experiments.crossfit.online_learning_from_logs sqlite|pg [n_seeds]
Output: results/crossfit/online_learning_from_logs_<engine>.{json,png}
"""
from __future__ import annotations
import json, sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.crossfit import online_learning as PQ
from plangate.paths import CROSSFIT_DIR

D = PQ.D
BLOCK, MIN_TRAIN, DEFAULT = PQ.BLOCK, PQ.MIN_TRAIN, PQ.DEFAULT
FAM = {"c0": "flash", "RESAMPLE_1": "flash", "RESAMPLE_2": "flash", "REPAIR_DB": "flash",
       "VALUE_FIX": "flash", "SWITCH_PRO": "pro", "SWITCH_QWEN": "qwen"}

# name -> config
REGIMES = {
    "static": dict(kind="static"),
    "gold-full (Q-V)": dict(kind="full", adv=False),
    "gold-full (ADV)": dict(kind="full", adv=True),
    "logs, gold 100% (Q-V)": dict(kind="logs", label="gold", beta=1.0, adv=False),
    "logs, gold 100% (ADV)": dict(kind="logs", label="gold", beta=1.0, adv=True),
    "logs, self-db (ADV)": dict(kind="logs", label="self_db", beta=0.0, adv=True),
    "logs, self-db + audit 10% unc (ADV)": dict(kind="logs", label="self_db", beta=0.1, mode="unc", adv=True),
    "logs, self-db + audit 25% unc (ADV)": dict(kind="logs", label="self_db", beta=0.25, mode="unc", adv=True),
    "logs, self-db + audit 25% rand (ADV)": dict(kind="logs", label="self_db", beta=0.25, mode="rand", adv=True),
}
EPS_EXTRA = 0.1


# ------------------------------------------------------------------ labels
def db_wrong(e):
    ev = e["ev"]; eb = ev["EXEC_BOUNDED"]
    return (not ev["PARSE_BIND"].get("valid", True) or not eb.get("ok") or eb.get("empty")
            or eb.get("all_null") or bool((ev.get("VALUE_PROBE") or {}).get("n_missing")))


def self_label_db(r, names):
    fams = defaultdict(set)
    for x in names:
        if not db_wrong(r[x]):
            fams[D.fp(r[x])].add(FAM[x])
    lab = {}
    for x in names:
        if db_wrong(r[x]):
            lab[x] = 0
        elif len(fams[D.fp(r[x])]) >= 2:
            lab[x] = 1
    return lab


# ------------------------------------------------------------------ advantage model
class CFReg:
    """Per-fold gradient-boosted regressor (fold i trained on train_folds[i])."""

    def fit(self, X, y, groups, folds):
        self.names = sorted(set().union(*X)) if X else []
        M = self._m(X); y = np.array(y, float); g = np.array(groups)
        self.models = []
        for tr in folds:
            idx = np.isin(g, tr)
            if idx.sum() < 20:
                self.models.append(float(y[idx].mean()) if idx.any() else 0.0); continue
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0)
            m.fit(M[idx], y[idx]); self.models.append(m)
        return self

    def _m(self, X):
        return np.array([[d.get(n, 0) for n in self.names] for d in X], dtype=float)

    def predict(self, X, fold_ids):
        M = self._m(X); out = np.zeros(len(X)); fold_ids = np.array(fold_ids)
        for i, m in enumerate(self.models):
            idx = fold_ids == i
            if idx.any():
                out[idx] = m if isinstance(m, float) else m.predict(M[idx])
        return out


def add_adv(P, rows, items, fold_of, train_folds):
    X, y, g, keys = [], [], [], []
    for (i, S, a) in P["Q"]:
        r = rows[i]
        best, ab, a2 = P["select"](i, S)
        f = D.state_feats(r, items[r["question_id"]], S, best, ab, a2, "full")
        for x in P["acts"]:
            f["act_" + x] = int(x == a)
        b2, _, _ = P["select"](i, P["canon"][(i, frozenset(S + (a,)))])
        X.append(f); y.append(int(r[b2]["correct"]) - int(r[best]["correct"])); g.append(i); keys.append((i, S, a))
    m = CFReg().fit(X, y, g, train_folds)
    P["ADV"] = dict(zip(keys, m.predict(X, [fold_of[i] for i in g])))
    return P


def run_hybrid(rows, P, lam, tau, adv, strong="SWITCH_QWEN", max_cheap=2):
    """Two-phase policy of crossfit.run_hybrid; with adv=True the cheap-action score is ADV(s,a) - lam*c."""
    if not adv:
        return D.run_hybrid(rows, P, "full", lam, tau, strong, max_cheap)
    out = []
    cheap = [a for a in P["acts"] if a != strong]
    for i, r in enumerate(rows):
        S = ("c0",); cost, lat = D.cand_cost(r, "c0", "full"); path = []
        while len(S) - 1 < max_cheap:
            best_a, best_score = None, 0.0
            for a in cheap:
                if a in S or (i, S, a) not in P["ADV"]:
                    continue
                c, l = D.cand_cost(r, a, "full")
                s = P["ADV"][(i, S, a)] - lam * c
                if s > best_score:
                    best_a, best_score = a, s
            if best_a is None:
                break
            c, l = D.cand_cost(r, best_a, "full"); cost += c; lat += l; path.append(best_a)
            S = P["canon"][(i, frozenset(S + (best_a,)))]
        if P["V"][(i, S)] < tau and D.available(r, strong):
            S2 = P["canon"].get((i, frozenset(S + (strong,))))
            if S2 is not None:
                c, l = D.cand_cost(r, strong, "full"); cost += c; lat += l; path.append(strong); S = S2
        g, _, _ = P["select"](i, S)
        out.append({"correct": int(r[g]["correct"]), "cost": cost, "lat": lat, "plan": "->".join(path) or "ACCEPT",
                    "final": g, "n_llm": 1 + sum(a != "VALUE_FIX" for a in path), "v_final": P["V"][(i, S)]})
    return out


# ------------------------------------------------------------------ streams
def stream_full(rows, items, adv):
    """Idealised: every past question with every action and gold labels."""
    n = len(rows); block = np.arange(n) // BLOCK; nb = block.max() + 1
    folds = [np.where(block < max(b, 1))[0] for b in range(nb)]   # block 0 is served by the rule
    P = D.build_planner(rows, items, "full", block, folds, "pairwise")
    if adv:
        add_adv(P, rows, items, block, folds)
    hyb = run_hybrid(rows, P, *DEFAULT, adv)
    return [dict(PQ.rule(r), explored=0) if block[i] == 0 else hyb[i] for i, r in enumerate(rows)], n, n


def stream_logs(rows, items, cfg, seed):
    n = len(rows); rng = np.random.default_rng(3000 + seed)
    log, res, P, n_audit = [], [None] * n, None, 0
    for b0 in range(0, n, BLOCK):
        blk = list(range(b0, min(n, b0 + BLOCK)))
        hyb = run_hybrid([x for _, x in log] + [rows[i] for i in blk], P, *DEFAULT, cfg["adv"])[len(log):] \
            if P is not None else None
        served = []
        for j, i in enumerate(blk):
            r = rows[i]
            if hyb is None:
                x = dict(PQ.rule(r), explored=0); names = x["plan"].split("+")
            else:
                x = dict(hyb[j], explored=0)
                names = ["c0"] + [a for a in x["plan"].split("->") if a in r and a != "c0"]
            if rng.random() < EPS_EXTRA:
                extra = [a for a in PQ.ALL_ACTS if D.available(r, a) and a not in names]
                if extra:
                    a = extra[rng.integers(len(extra))]
                    x = dict(x, cost=x["cost"] + D.cand_cost(r, a, "full")[0]); names = names + [a]
            res[i] = x; served.append((i, names, x.get("v_final")))
        # feedback for this block
        k = int(round(cfg["beta"] * len(blk)))
        if cfg["beta"] >= 1:
            audited = {i for i, _, _ in served}
        elif k == 0:
            audited = set()
        elif cfg.get("mode") == "unc" and hyb is not None:
            audited = {i for i, _, v in sorted(served, key=lambda t: abs((t[2] if t[2] is not None else 0.5) - 0.5))[:k]}
        else:
            audited = {served[t][0] for t in rng.choice(len(served), k, replace=False)}
        n_audit += len(audited)
        for i, names, _ in served:
            r = rows[i]
            if i in audited:
                lab = {g: int(r[g]["correct"]) for g in names}
            else:
                lab = self_label_db(r, names) if cfg["label"] == "self_db" else {}
            if "c0" not in lab:
                continue
            log.append((i, PQ.masked(r, [g for g in names if g in lab], lab)))
        if len(log) >= MIN_TRAIN and b0 + BLOCK < n:
            sub = [x for _, x in log] + [rows[i] for i in range(b0 + BLOCK, min(n, b0 + 2 * BLOCK))]
            fold = np.zeros(len(sub), int); tr = [np.arange(len(log))]
            P = D.build_planner(sub, items, "full", fold, tr, "pairwise")
            if cfg["adv"]:
                add_adv(P, sub, items, fold, tr)
    return res, len(log), n_audit


def summ(res, idx):
    sub = [res[i] for i in idx]
    return float(np.mean([x["correct"] for x in sub])), float(np.mean([x["cost"] for x in sub]))


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    only = sys.argv[3].split("|") if len(sys.argv) > 3 else list(REGIMES)
    base, items = D.load_action_table(engine)
    n = len(base); nb = (n + BLOCK - 1) // BLOCK
    late = list(range(n - n // 3, n))
    print(f"== Online learning from logs ({engine}, n={n}, block={BLOCK}, seeds={n_seeds}, point lam,tau={DEFAULT})", flush=True)
    per = defaultdict(list)   # name -> per seed dict
    for seed in range(n_seeds):
        perm = np.random.default_rng(seed).permutation(n)
        rows = [base[j] for j in perm]
        for name in only:
            cfg = REGIMES[name]
            if cfg["kind"] == "static":
                res, nl, na = [dict(PQ.rule(r), explored=0) for r in rows], 0, 0
            elif cfg["kind"] == "full":
                res, nl, na = stream_full(rows, items, cfg["adv"])
            else:
                res, nl, na = stream_logs(rows, items, cfg, seed)
            d = {"all": summ(res, range(n)), "late": summ(res, late), "logged": nl, "audited": na,
                 "curve": [summ(res, range(b * BLOCK, min(n, (b + 1) * BLOCK))) for b in range(nb)],
                 "late_correct": [res[i]["correct"] for i in late],
                 "accept_late": float(np.mean([res[i]["plan"] in ("ACCEPT", "c0") for i in late]))}
            per[name].append(d)
            print(f"  seed {seed} {name:38s} logged={nl:3d} audited={na:3d}  all {d['all'][0]:.3f}/{d['all'][1]:5.2f}"
                  f"  late {d['late'][0]:.3f}/{d['late'][1]:5.2f}  accept={d['accept_late']:.2f}", flush=True)

    print("\n== Mean over seeds: last third of the stream (after learning)")
    print(f"  {'regime':38s} {'EX':>6} {'cost':>6} {'ACCEPT':>7} {'labels/q':>8}  vs static (paired, 95% CI)")
    rep = {"engine": engine, "n": n, "block": BLOCK, "point": DEFAULT, "regimes": {}}
    rng = np.random.default_rng(0)
    st = np.array([d["late_correct"] for d in per["static"]]) if "static" in per else None
    for name, ds in per.items():
        a = np.mean([d["late"] for d in ds], 0)
        lab = np.mean([d["audited"] for d in ds]) / n
        line = f"  {name:38s} {a[0]:6.3f} {a[1]:6.2f} {np.mean([d['accept_late'] for d in ds]):7.2f} {lab:8.2f}"
        ci = None
        if st is not None and name != "static":
            diff = (np.array([d["late_correct"] for d in ds]) - st).mean(0)   # per question, averaged over orders
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)]
            ci = [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            line += f"  {100*ci[0]:+5.1f} [{100*ci[1]:+5.1f}, {100*ci[2]:+5.1f}]"
        print(line)
        rep["regimes"][name] = {"late": a.tolist(), "all": np.mean([d["all"] for d in ds], 0).tolist(),
                                "accept_late": float(np.mean([d["accept_late"] for d in ds])),
                                "labels_per_q": float(lab), "vs_static": ci,
                                "curve": np.mean([d["curve"] for d in ds], 0).tolist()}
    p = CROSSFIT_DIR / f"online_learning_from_logs_{engine}.json"
    old = json.loads(p.read_text(encoding="utf8")) if p.exists() and len(only) < len(REGIMES) else {"regimes": {}}
    old["regimes"].update(rep["regimes"]); rep["regimes"] = old["regimes"]
    p.write_text(json.dumps(rep, indent=1), encoding="utf8")
    plot(rep, engine, nb)


def plot(rep, engine, nb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = rep["regimes"]
    if "static" not in R:
        return
    show = {"gold-full (ADV)": ("#d62728", "-"), "logs, gold 100% (Q-V)": ("#1f77b4", ":"),
            "logs, gold 100% (ADV)": ("#1f77b4", "-"), "logs, self-db (ADV)": ("#2ca02c", "-"),
            "logs, self-db + audit 10% unc (ADV)": ("#ff7f0e", "--"),
            "logs, self-db + audit 25% unc (ADV)": ("#ff7f0e", "-"),
            "logs, self-db + audit 25% rand (ADV)": ("#9467bd", "-.")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(nb) * rep["block"] + rep["block"] / 2
    w = np.minimum(rep["block"], rep["n"] - np.arange(nb) * rep["block"])
    st = np.array(R["static"]["curve"])
    ax[0].axhline(0, color="k", ls=":", lw=1, label="static")
    ax[1].plot(x, st[:, 1], "k:", lw=1, label="static")
    for name, (c, ls) in show.items():
        if name not in R:
            continue
        cv = np.array(R[name]["curve"])
        ax[0].plot(x, 100 * np.cumsum((cv[:, 0] - st[:, 0]) * w) / np.cumsum(w), color=c, ls=ls, label=name)
        ax[1].plot(x, cv[:, 1], color=c, ls=ls)
    ax[0].set_xlabel("questions served"); ax[0].set_ylabel("cumulative EX gain over static (points)")
    ax[1].set_xlabel("questions served"); ax[1].set_ylabel("cost per question in block"); ax[1].set_ylim(0, 12)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    path = CROSSFIT_DIR / f"online_learning_from_logs_{engine}.png"
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print("saved", path)


if __name__ == "__main__":
    main()
