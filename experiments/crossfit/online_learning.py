"""Online learning: a self-improving planner over a query stream (prequential evaluation).

Questions arrive one by one in a random order. The planner serving block b is trained only on
the questions of blocks < b (no look-ahead, no cross-fitting on the evaluation set).
Everything is replayed from the cached action table, so no LLM call is made.

Feedback regimes (what the system can learn from):
  static       no learning: the rule cascade (DB trigger -> strong model), for the whole stream
  gold-full    every past question, every action, gold labels          (idealised upper bound)
  explore+gold epsilon-exploration: an explored question runs ALL actions (paid) and is answered
               by execution vote; only explored questions are logged, with gold labels (sparse audit)
  explore+self same, but labels are self-generated: a candidate is labelled correct iff its result
               fingerprint equals the plurality result cluster of the explored question (no gold)
The first block is always a bootstrap block (everything explored) for the explore regimes.

L1 evidence store: DB evidence keyed by (db, normalised SQL) and value probes keyed by
(db, table, column, op, literal) are reused across the stream; reports hit rates and DB time saved.

Usage:  python -m experiments.crossfit.online_learning sqlite|pg [n_seeds]
Output: results/crossfit/online_learning_<engine>.{json,png} (stdout: online_learning_<engine>.txt)
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict

import numpy as np

from plangate.features import normalize_sql
from plangate.paths import CROSSFIT_DIR
from plangate.planning import crossfit as D
from plangate.planning.crossfit import triggers

BLOCK = 30
EPS = (0.1, 0.2, 0.3)
POINTS = [(lam, tau) for lam in (0.02, 0.05, 0.1) for tau in (0.3, 0.4, 0.5, 0.6, 0.7)]
DEFAULT = (0.05, 0.5)
MIN_TRAIN = 8          # fewer logged questions than this -> keep serving with the rule cascade
# learning from own logs: name -> (label source, bootstrap questions, extra-exploration probability)
LOGGED = {"logs, gold": ("gold", 0, 0.1), "logs, self": ("self", 0, 0.1),
          "boot30+logs, gold": ("gold", 30, 0.1), "boot30+logs, self": ("self", 30, 0.1)}


ALL_ACTS = ["VALUE_FIX", "RESAMPLE_1", "RESAMPLE_2", "REPAIR_DB", "SWITCH_PRO", "SWITCH_QWEN"]


def explore_set(r):
    return ["c0"] + [a for a in ALL_ACTS if D.available(r, a)]


def pseudo_rows(rows):
    """Self-labelling: correct := result equals the plurality execution cluster of all candidates.
    Ties are broken toward the cluster containing the strongest model."""
    out, agree, n = [], 0, 0
    pref = ["SWITCH_QWEN", "SWITCH_PRO", "c0", "RESAMPLE_1", "RESAMPLE_2", "REPAIR_DB", "VALUE_FIX"]
    for r in rows:
        names = explore_set(r)
        cnt = Counter(D.fp(r[x]) for x in names if D.fp(r[x]))
        top = None
        if cnt:
            m = max(cnt.values())
            for x in pref:
                if x in names and D.fp(r[x]) and cnt[D.fp(r[x])] == m:
                    top = D.fp(r[x]); break
        r2 = dict(r)
        for x in names:
            lab = int(top is not None and D.fp(r[x]) == top)
            r2[x] = dict(r[x], correct=lab)
            agree += lab == int(r[x]["correct"]); n += 1
        out.append(r2)
    return out, agree / n


def rule(r):
    """Cold-start / static policy: DB trigger on c0 -> strong model (replace)."""
    if triggers(r["c0"]["ev"]) and D.available(r, "SWITCH_QWEN"):
        return D.res(r, "SWITCH_QWEN", ["c0", "SWITCH_QWEN"])
    return D.res(r, "c0", ["c0"])


def explore_answer(r):
    names = explore_set(r)
    return dict(D.res(r, D.vote(r, names), names, "full"), explored=1)


def stream(rows, items, regime, eps, seed, label_rows):
    """Returns {point: per-question results in stream order} for one regime and one stream order."""
    n = len(rows)
    rng = np.random.default_rng(1000 + seed)
    block = np.arange(n) // BLOCK
    nb = block.max() + 1
    if regime == "gold-full":
        explored = np.zeros(n, bool)
        logged = np.ones(n, bool)
    elif regime == "static":
        explored = np.zeros(n, bool); logged = np.zeros(n, bool)
    else:
        explored = (block == 0) | (rng.random(n) < eps)
        logged = explored
    train_folds = [np.where(logged & (block < b))[0] for b in range(nb)]
    trainable = [len(t) >= MIN_TRAIN for t in train_folds]
    first = next((b for b in range(nb) if trainable[b]), None)
    P = None
    if regime != "static" and first is not None:
        tf = [train_folds[b] if trainable[b] else train_folds[first] for b in range(nb)]
        P = D.build_planner(label_rows, items, "full", block, tf, "pairwise")
    out = {}
    for pt in POINTS:
        lam, tau = pt
        hyb = D.run_hybrid(rows, P, "full", lam, tau) if P is not None else None
        res = []
        for i, r in enumerate(rows):
            if explored[i]:
                res.append(explore_answer(r))
            elif P is None or not trainable[block[i]]:
                res.append(dict(rule(r), explored=0))
            else:
                res.append(dict(hyb[i], explored=0))
        out[pt] = res
    return out, int(logged.sum()), int(explored.sum())


def self_label(r, names):
    """Label the acquired candidates of one logged question without gold.
    None if fewer than two candidates executed (nothing to compare)."""
    pref = ["SWITCH_QWEN", "SWITCH_PRO", "c0", "RESAMPLE_1", "RESAMPLE_2", "REPAIR_DB", "VALUE_FIX"]
    ok = [x for x in names if D.fp(r[x])]
    if len(ok) < 2:
        return None
    cnt = Counter(D.fp(r[x]) for x in ok)
    m = max(cnt.values())
    top = next(D.fp(r[x]) for x in pref if x in ok and cnt[D.fp(r[x])] == m)
    return {x: int(D.fp(r[x]) == top) for x in names}


def masked(r, names, labels=None):
    """The log record of a served question: only the acquired candidates are visible."""
    r2 = {k: v for k, v in r.items() if k not in ALL_ACTS + ["c0", "REPAIR_BLIND"]}
    for x in names:
        r2[x] = dict(r[x], correct=labels[x]) if labels else r[x]
    for x in ALL_ACTS + ["REPAIR_BLIND"]:
        r2.setdefault(x, None)
    return r2


def stream_logged(rows, items, label, boot, eps_extra, seed, point=DEFAULT):
    """Learning from the system's own logs, retrained after every block.
    boot      : number of bootstrap questions answered by explore-all (paid)
    eps_extra : probability of acquiring one extra random action after answering (paid, logged only)
    label     : 'gold' (delayed correctness feedback on acquired candidates) or 'self'"""
    n = len(rows)
    rng = np.random.default_rng(2000 + seed)
    log = []                     # (stream index, masked labelled row)
    res = [None] * n
    P, n_logged = None, 0
    for b0 in range(0, n, BLOCK):
        blk = list(range(b0, min(n, b0 + BLOCK)))
        if P is not None:
            sub = [x for _, x in log] + [rows[i] for i in blk]
            hyb = D.run_hybrid(sub, P, "full", *point)[len(log):]
        for j, i in enumerate(blk):
            r = rows[i]
            if i < boot:
                x = explore_answer(r); names = explore_set(r)
            elif P is None:
                x = dict(rule(r), explored=0); names = x["plan"].split("+")
            else:
                x = dict(hyb[j], explored=0)
                names = ["c0"] + [a for a in x["plan"].split("->") if a in r and a != "c0"]
            if eps_extra and i >= boot and rng.random() < eps_extra:
                extra = [a for a in ALL_ACTS if D.available(r, a) and a not in names]
                if extra:
                    a = extra[rng.integers(len(extra))]
                    c, l = D.cand_cost(r, a, "full")
                    x = dict(x, cost=x["cost"] + c); names = names + [a]
            res[i] = x
            labs = ({g: int(r[g]["correct"]) for g in names} if label == "gold" else self_label(r, names))
            if labs is not None:
                log.append((i, masked(r, names, labs)))
        if len(log) >= MIN_TRAIN and b0 + BLOCK < n:
            nxt = [rows[i] for i in range(b0 + BLOCK, min(n, b0 + 2 * BLOCK))]
            sub = [x for _, x in log] + nxt
            fold = np.zeros(len(sub), int)
            P = D.build_planner(sub, items, "full", fold, [np.arange(len(log))], "pairwise")
            n_logged = len(log)
    return res, n_logged


def offline_crossfit(rows, items):
    """Reference: the cross-fitted planner of experiments/crossfit/planner.py (trained on the other 4/5)."""
    fold_of, train_folds = D.question_folds(len(rows))
    P = D.build_planner(rows, items, "full", fold_of, train_folds, "pairwise")
    return {pt: D.run_hybrid(rows, P, "full", *pt) for pt in POINTS}


def evidence_store(rows, results):
    """L1: DB evidence reuse along the stream, for the candidates the policy actually acquired."""
    seen_sql, seen_vp = set(), set()
    tot = hit = 0.0; n_sql = h_sql = n_vp = h_vp = 0
    for r, x in zip(rows, results):
        if x.get("explored"):
            names = explore_set(r)
        else:   # planner plans look like "VALUE_FIX->SWITCH_QWEN", rule cascade plans like "c0+SWITCH_QWEN"
            names = ["c0"] + [a for a in x["plan"].replace("+", "->").split("->") if a in r and a != "c0"]
        for g in names:
            e = r.get(g)
            if not e:
                continue
            k = (r["db_id"], normalize_sql(e["sql"]))
            ms_sql = sum(e["db_ms"].get(op, 0) for op in ("PARSE_BIND", "EXPLAIN", "EXEC_BOUNDED"))
            n_sql += 1; tot += ms_sql
            if k in seen_sql:
                h_sql += 1; hit += ms_sql
            seen_sql.add(k)
            vp = e["ev"].get("VALUE_PROBE") or {}
            probes = vp.get("probes") or []
            ms_vp = e["db_ms"].get("VALUE_PROBE", 0)
            tot += ms_vp
            for p in probes:
                kv = (r["db_id"], p.get("table"), p.get("column"), p.get("op"), str(p.get("literal")).lower())
                n_vp += 1
                if kv in seen_vp:
                    h_vp += 1; hit += ms_vp / len(probes)
                seen_vp.add(kv)
    return {"sql_hit": h_sql / max(n_sql, 1), "vp_hit": h_vp / max(n_vp, 1),
            "db_ms_saved_frac": hit / max(tot, 1e-9), "db_ms_per_q": tot / len(rows),
            "db_ms_saved_per_q": hit / len(rows)}


def summ(res, idx=None):
    idx = range(len(res)) if idx is None else idx
    sub = [res[i] for i in idx]
    return float(np.mean([x["correct"] for x in sub])), float(np.mean([x["cost"] for x in sub]))


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    base, items = D.load_action_table(engine)
    n = len(base); nb = (n + BLOCK - 1) // BLOCK
    regimes = ([("static", 0), ("gold-full", 0), ("explore+self", 0.1)]
               + [(k, 0) for k in LOGGED])

    print(f"== Online learning ({engine}, n={n}, block={BLOCK}, seeds={n_seeds})", flush=True)
    off = offline_crossfit(base, items)
    rep = {"engine": engine, "n": n, "block": BLOCK, "points": POINTS, "default": DEFAULT,
           "offline": {str(pt): summ(v) for pt, v in off.items()}, "regimes": {}}
    flash_cost = float(np.mean([r["c0"]["cost"] for r in base]))
    rep["flash_cost"] = flash_cost

    acc = defaultdict(lambda: defaultdict(list))     # regime -> point -> [(acc, cost) per seed]
    curve = defaultdict(list)                        # regime -> per seed: per-block (acc, cost) at DEFAULT
    late = defaultdict(lambda: defaultdict(list))    # last third of the stream
    l1 = defaultdict(list); plab = []; nlog = defaultdict(list)
    for seed in range(n_seeds):
        perm = np.random.default_rng(seed).permutation(n)
        rows = [base[j] for j in perm]
        prow, pa = pseudo_rows(rows); plab.append(pa)
        for reg, eps in regimes:
            name = reg if not eps else f"{reg} eps={eps}"
            if reg in LOGGED:
                lab, boot, ex = LOGGED[reg]
                r0, nl = stream_logged(rows, items, lab, boot, ex, seed)
                out, ne = {DEFAULT: r0}, sum(x.get("explored", 0) for x in r0)
            else:
                lr = prow if reg == "explore+self" else rows
                out, nl, ne = stream(rows, items, reg, eps, seed, lr)
            nlog[name].append((nl, ne))
            lt = range(n - n // 3, n)
            for pt, res in out.items():
                acc[name][pt].append(summ(res)); late[name][pt].append(summ(res, lt))
            res = out[DEFAULT]
            curve[name].append([summ(res, range(b * BLOCK, min(n, (b + 1) * BLOCK))) for b in range(nb)])
            l1[name].append(evidence_store(rows, res))
            print(f"  seed {seed} {name:24s} logged={nl:3d} explored={ne:3d}  "
                  f"acc={summ(res)[0]:.3f} cost={summ(res)[1]:5.2f}  last third acc={summ(res, lt)[0]:.3f} "
                  f"cost={summ(res, lt)[1]:5.2f}", flush=True)

    print(f"\n  self-label agreement with gold (candidate level): {np.mean(plab):.3f}")
    print(f"\n== Mean over seeds at default point lam={DEFAULT[0]}, tau={DEFAULT[1]}  (offline cross-fit: "
          f"acc {rep['offline'][str(DEFAULT)][0]:.3f}, cost {rep['offline'][str(DEFAULT)][1]:.2f})")
    print(f"  {'regime':24s} {'acc':>6} {'cost':>6} | {'last1/3 acc':>11} {'cost':>6} | {'L1 sql hit':>10} "
          f"{'vp hit':>6} {'DB saved':>8}")
    for name in acc:
        a = np.mean(acc[name][DEFAULT], 0); b = np.mean(late[name][DEFAULT], 0)
        e = {k: np.mean([x[k] for x in l1[name]]) for k in l1[name][0]}
        print(f"  {name:24s} {a[0]:6.3f} {a[1]:6.2f} | {b[0]:11.3f} {b[1]:6.2f} | {e['sql_hit']:10.2f} "
              f"{e['vp_hit']:6.2f} {e['db_ms_saved_frac']:8.0%}")
        rep["regimes"][name] = {
            "logged_explored": nlog[name],
            "points": {str(pt): {"all": np.mean(acc[name][pt], 0).tolist(), "last_third": np.mean(late[name][pt], 0).tolist(),
                                 "all_sd": np.std([x[0] for x in acc[name][pt]]).item()}
                       for pt in acc[name]},
            "curve": np.mean(curve[name], 0).tolist(), "l1": e}

    # matched-budget view over the operating-point grid (last third of the stream = after learning)
    print("\n== Last third of the stream: best accuracy within a cost budget (grid of (lam,tau) points)")
    for mult in (1.25, 1.5, 2.0, 3.0):
        B = flash_cost * mult
        line = f"  {mult:.2f}x ({B:5.2f}):"
        for name in acc:
            pts = [np.mean(late[name][pt], 0) for pt in late[name]]
            ok = [p for p in pts if p[1] <= B]
            line += f"  {name}={max(p[0] for p in ok):.3f}" if ok else f"  {name}=  -  "
        print(line)
    rep["self_label_agreement"] = float(np.mean(plab))
    (CROSSFIT_DIR / f"online_learning_{engine}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf8")
    plot(rep, engine, nb)


def plot(rep, engine, nb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    show = ["static", "gold-full", "logs, gold", "logs, self", "boot30+logs, self", "explore+self eps=0.1"]
    style = {"static": ("k", ":"), "gold-full": ("#d62728", "-"), "logs, gold": ("#1f77b4", "-"),
             "logs, self": ("#2ca02c", "-"), "boot30+logs, self": ("#2ca02c", "--"),
             "explore+self eps=0.1": ("#9467bd", ":")}
    x = np.arange(nb) * rep["block"] + rep["block"] / 2
    w = np.minimum(rep["block"], rep["n"] - np.arange(nb) * rep["block"])
    st = np.array(rep["regimes"]["static"]["curve"])
    for name in show:
        if name not in rep["regimes"]:
            continue
        c = np.array(rep["regimes"][name]["curve"])
        # cumulative accuracy gain over the static policy on the same question order (removes the
        # difficulty of each stretch of the stream), and cost per question in each block
        gain = np.cumsum((c[:, 0] - st[:, 0]) * w) / np.cumsum(w)
        ax[0].plot(x, 100 * gain, color=style[name][0], ls=style[name][1], label=name)
        ax[1].plot(x, c[:, 1], color=style[name][0], ls=style[name][1], label=name)
    off = rep["offline"][str(tuple(rep["default"]))] if str(tuple(rep["default"])) in rep["offline"] else None
    if off:
        ax[1].axhline(off[1], color="gray", lw=0.8, ls="-.", label="offline cross-fit (reference)")
    ax[1].set_ylim(0, 12)
    ax[0].set_xlabel("questions served"); ax[0].set_ylabel("cumulative EX gain over static (points)")
    ax[1].set_xlabel("questions served"); ax[1].set_ylabel("cost per question in block")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = CROSSFIT_DIR / f"online_learning_{engine}.png"
    fig.savefig(p, dpi=140); print("saved", p)


if __name__ == "__main__":
    main()
