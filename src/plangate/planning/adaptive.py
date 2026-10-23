"""Adaptive evidence acquisition: the PlanGate planner over partially observed DB evidence.

Optional DB evidence (VALUE_PROBE, BOUNDED_EXEC) becomes an explicit, priced action on an
individual candidate. PARSE_BIND and EXPLAIN stay mandatory (charged). The state records, per
candidate, which optional observations were acquired; features of unacquired observations are
*missing* (explicit has_* flags), never "clean".

Learned components (gradient-boosted trees, trained only on the training split):
  A(g|s)   acceptor over a candidate in a partially observed state
  P(a>b)   pairwise preference between selection clusters (unexecuted candidates are singletons)
  V(s)     Pr[SELECT(s) correct]
  Q(s,a)   Pr[SELECT(s + a) correct] for an LLM / rewrite action a (new candidate unobserved)
  G(s,e;l) E_o[ U_l(s + o) ] - U_l(s), the value of information of evidence action e, with the
           continuation value U_l(s) = max( V(s), max_a Q(s,a) - l*C(a) )
The policy buys e iff G(s,e;l) > l*C(e). For calibrated V and Q, G >= 0 (U is a max of affine
functions of the posterior, Jensen), so a probe is worth buying only when its outcome can change
the next decision by more than its price.

Policies (identical model pool, LLM outcomes, selection and cost accounting):
  cheap_only, no_db, fixed_bundle, optimizer_gated, rule_selective, adaptive, oracle
"""
from __future__ import annotations
import math, random
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

from ..features import p0_features, sql_features, db_features, normalize_sql

VP, EX = "VALUE_PROBE", "EXEC_BOUNDED"
OPT = (VP, EX)
MAND = ("PARSE_BIND", "EXPLAIN")
LLM_ACTS = ("VALUE_FIX", "RESAMPLE_1", "REPAIR", "SWITCH_PRO", "SWITCH_QWEN")
SRC = {"c0": "flash", "VALUE_FIX": "valuefix", "RESAMPLE_1": "resample", "REPAIR": "repair",
       "SWITCH_PRO": "pro", "SWITCH_QWEN": "qwen"}
MAX_LLM = 3
# REPAIR variants (visible-evidence masks on c0) built in the replay tables; SQLite uses {"vpex"} only
REPAIR_MASKS = ("none", "vp", "ex", "vpex")
LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35)
# optimizer-derived features (EXPLAIN cost / rows / operator counts), hidden in the no-optimizer ablation
OPT_KEYS = ("ex_log_cost", "ex_log_rows", "ex_est_one_row", "ex_seq_scans", "ex_index_scans", "ex_nested_loops",
            "ex_hash_joins", "ex_sorts", "ex_aggregates", "ex_unconditioned_nl")
# many masked states per question are near-duplicates: strong leaf regularization limits overfitting
EST_PARAMS = dict(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
                  min_samples_leaf=60, random_state=0)


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def mask_name(m):
    return {frozenset(): "none", frozenset({VP}): "vp", frozenset({EX}): "ex", frozenset({VP, EX}): "vpex"}[frozenset(m)]


# =============================================================================== state
class State:
    """cands: tuple of (action_key, repair_variant or None); masks: tuple of frozensets."""
    __slots__ = ("q", "cands", "masks")

    def __init__(self, q, cands, masks):
        self.q, self.cands, self.masks = q, tuple(cands), tuple(frozenset(m) for m in masks)

    def key(self):
        return (self.q, self.cands, self.masks)

    def add_obs(self, i, op):
        m = list(self.masks); m[i] = m[i] | {op}
        return State(self.q, self.cands, m)

    def add_cand(self, key, variant=None):
        return State(self.q, self.cands + ((key, variant),), self.masks + (frozenset(),))

    def llm_done(self):
        return [k for k, _ in self.cands[1:]]


def entry(row, cand):
    k, v = cand
    if k == "REPAIR":
        return row["REPAIR"][v]
    return row[k]


def fp(e):
    eb = e["ev"].get(EX) or {}
    return eb.get("fingerprint") if eb.get("ok") else None


def visible_ev(e, mask):
    return {op: e["ev"][op] for op in e["ev"] if op in MAND or op in mask}


def vis_fp(e, mask):
    return fp(e) if EX in mask else None


def visible_trigger(e, mask):
    ev = visible_ev(e, mask)
    pb, ex, vp, eb = (ev.get(k) for k in ("PARSE_BIND", "EXPLAIN", VP, EX))
    t = bool(pb is not None and not pb.get("valid", True))
    t |= bool(ex and ex.get("joins_without_on"))
    t |= bool(vp and vp.get("n_missing"))
    t |= bool(eb and (not eb.get("ok") or eb.get("empty") or eb.get("all_null")))
    return int(t)


def legal_llm(row, s, repair_allowed):
    done = set(s.llm_done())
    if len(done) >= MAX_LLM:
        return []
    out = []
    for a in LLM_ACTS:
        if a in done:
            continue
        if a == "VALUE_FIX" and (VP not in s.masks[0] or row.get("VALUE_FIX") is None):
            continue   # the rewrite exists only once VALUE_PROBE on c0 has been acquired
        if a == "REPAIR" and mask_name(s.masks[0]) not in repair_allowed:
            continue   # repair is conditioned on the evidence visible on c0; only supported masks
        out.append(a)
    return out


def legal_ev(s):
    return [(i, op) for i in range(len(s.cands)) for op in OPT if op not in s.masks[i]]


def apply_llm(s, a):
    return s.add_cand(a, mask_name(s.masks[0]) if a == "REPAIR" else None)


# =============================================================================== features
class Feat:
    """Feature extraction over a replay table. `no_opt` hides every optimizer-derived feature."""

    def __init__(self, rows, items, no_opt=False):
        self.no_opt = no_opt
        self.rows = {r["question_id"]: r for r in rows}
        self.p0 = {q: p0_features(items[q]) for q in self.rows}
        self.sqlf = {}

    def _sqlf(self, sql):
        if sql not in self.sqlf:
            self.sqlf[sql] = sql_features(sql)
        return self.sqlf[sql]

    def cand(self, s, i):
        row = self.rows[s.q]
        e, m = entry(row, s.cands[i]), s.masks[i]
        f = dict(self.p0[s.q])
        f.update(self._sqlf(e["sql"]))
        for src in set(SRC.values()):
            f["src_" + src] = int(SRC[s.cands[i][0]] == src)
        f.update(db_features(visible_ev(e, m)))
        if self.no_opt:
            for k in OPT_KEYS:
                f.pop(k, None)
        f["is_vp"], f["is_exec"] = int(VP in m), int(EX in m)
        f["trig"] = visible_trigger(e, m)
        ns = normalize_sql(e["sql"])
        others = [j for j in range(len(s.cands)) if j != i]
        f["n_cands"] = len(s.cands)
        f["sql_agree_n"] = sum(normalize_sql(entry(row, s.cands[j])["sql"]) == ns for j in others)
        fps = [vis_fp(entry(row, s.cands[j]), s.masks[j]) for j in range(len(s.cands))]
        mine = fps[i]
        execd = [x for x in fps if x is not None]
        cl = Counter(execd)
        f["n_exec"] = sum(EX in s.masks[j] for j in range(len(s.cands)))
        f["n_unexec"] = len(s.cands) - f["n_exec"]
        f["res_agree_n"] = (cl[mine] - 1) if mine else 0
        f["res_disagree_n"] = (len(execd) - cl[mine]) if mine else 0
        f["is_plurality"] = int(bool(mine) and cl[mine] == max(cl.values()))
        f["n_clusters"] = len(cl)
        return f

    def state(self, s, best, a_best, a_second):
        f = {"st_" + k: v for k, v in self.cand(s, best).items()}
        c0 = self.cand(s, 0)
        for k in ("is_vp", "is_exec", "trig", "vp_missing", "eb_empty", "eb_fail", "res_agree_n",
                  "res_disagree_n", "ex_log_cost", "sql_literals"):
            f["c0_" + k] = c0.get(k, 0)
        f["best_is_c0"] = int(best == 0)
        f["A_best"], f["A_second"], f["A_margin"] = a_best, a_second, a_best - a_second
        done = set(s.llm_done())
        for a in LLM_ACTS:
            f["done_" + a] = int(a in done)
        f["n_llm"] = len(done)
        row = self.rows[s.q]
        f["vf_avail"] = int(VP in s.masks[0] and row.get("VALUE_FIX") is not None)
        return f

    def pair(self, fa, fb, aa, ab):
        f = {"d_" + k: fa.get(k, 0) - fb.get(k, 0) for k in set(fa) | set(fb)}
        for k in [x for x in fa if x.startswith("src_")] + ["is_exec", "is_vp", "trig"]:
            f["a_" + k], f["b_" + k] = fa.get(k, 0), fb.get(k, 0)
        f["Aa"], f["Ab"] = aa, ab
        return f


# =============================================================================== estimators
class Est:
    """Gradient-boosted estimator with per-fold models (for out-of-fold predictions) and a model
    fitted on all training questions (for serving)."""

    def __init__(self, kind="clf"):
        self.kind = kind

    def _new(self):
        p = dict(EST_PARAMS)
        if self.kind == "clf":
            return HistGradientBoostingClassifier(**p)
        return HistGradientBoostingRegressor(**p)

    def mat(self, X):
        return np.array([[d.get(n, 0.0) for n in self.names] for d in X], dtype=float)

    def fit(self, X, y, groups, folds):
        self.names = sorted(set().union(*X))
        M, y, g = self.mat(X), np.asarray(y, float), np.asarray(groups)
        self.models = []
        for tr in folds:
            idx = np.isin(g, tr)
            self.models.append(self._fit1(M[idx], y[idx]))
        self.full = self._fit1(M, y)
        return self

    def _fit1(self, M, y):
        if self.kind == "clf" and len(set(y.tolist())) < 2:
            return float(y.mean()) if len(y) else 0.5
        if len(y) == 0:
            return 0.0
        return self._new().fit(M, y)

    def predict(self, X, fold_ids):
        """fold_ids: per row fold index, or -1 for the full model."""
        if not X:
            return np.zeros(0)
        M = self.mat(X); out = np.zeros(len(X)); fold_ids = np.asarray(fold_ids)
        for fid in set(fold_ids.tolist()):
            m = self.full if fid < 0 else self.models[fid]
            idx = fold_ids == fid
            if isinstance(m, float):
                out[idx] = m
            elif self.kind == "clf":
                out[idx] = m.predict_proba(M[idx])[:, 1]
            else:
                out[idx] = m.predict(M[idx])
        return out


# =============================================================================== selection
def select_batch(states, F, A, P, fold_of):
    """Verified selection on partially observed states. Returns list of (best_idx, a_best, a_second)."""
    cf, cidx = [], []
    for si, s in enumerate(states):
        for i in range(len(s.cands)):
            cf.append(F.cand(s, i)); cidx.append((si, i))
    av = A.predict(cf, [fold_of(states[si].q) for si, _ in cidx])
    Aval = defaultdict(dict)
    feat = defaultdict(dict)
    for (si, i), a, f in zip(cidx, av, cf):
        Aval[si][i] = a; feat[si][i] = f
    reps, pf, pidx = {}, [], []
    for si, s in enumerate(states):
        row = F.rows[s.q]
        cl = defaultdict(list)
        for i in range(len(s.cands)):
            k = vis_fp(entry(row, s.cands[i]), s.masks[i])
            cl[("fp", k) if k else ("u", i)].append(i)
        rp = [max(g, key=lambda i: Aval[si][i]) for g in cl.values()]
        reps[si] = rp
        if len(rp) > 1:
            for a in rp:
                for b in rp:
                    if a != b:
                        pf.append(F.pair(feat[si][a], feat[si][b], Aval[si][a], Aval[si][b])); pidx.append((si, a, b))
    pv = P.predict(pf, [fold_of(states[si].q) for si, _, _ in pidx]) if pf else []
    Pval = {k: v for k, v in zip(pidx, pv)}
    out = []
    for si, s in enumerate(states):
        rp = reps[si]
        if len(rp) == 1:
            g = rp[0]
        else:
            sc = {a: sum(Pval[(si, a, b)] - Pval[(si, b, a)] for b in rp if b != a) for a in rp}
            g = max(rp, key=lambda a: (sc[a], Aval[si][a]))
        others = [Aval[si][i] for i in range(len(s.cands)) if i != g]
        out.append((g, Aval[si][g], max(others) if others else 0.0))
    return out


# =============================================================================== training
def sample_states(row, rng, repair_allowed, n_masks=4):
    q = row["question_id"]
    avail = [a for a in LLM_ACTS if (a != "VALUE_FIX" or row.get("VALUE_FIX") is not None)]
    states = []
    allm = [frozenset(), frozenset({VP}), frozenset({EX}), frozenset({VP, EX})]
    for k in range(MAX_LLM + 1):
        for T in combinations(avail, k):
            n = len(T) + 1
            masks_list = [[frozenset()] * n, [frozenset({VP, EX})] * n]
            for _ in range(n_masks if k <= 2 else 2):
                masks_list.append([rng.choice(allm) for _ in range(n)])
            for masks in masks_list:
                masks = list(masks)
                if "VALUE_FIX" in T:
                    masks[0] = masks[0] | {VP}
                cands = [("c0", None)]
                for a in T:
                    v = None
                    if a == "REPAIR":
                        # repair variant: c0's evidence visible when REPAIR was invoked (subset of c0's mask)
                        okm = [m for m in allm if m <= masks[0] and mask_name(m) in repair_allowed]
                        if not okm:
                            m = rng.choice([m for m in allm if mask_name(m) in repair_allowed])
                            masks[0] = masks[0] | m; okm = [m]
                        v = mask_name(rng.choice(okm))
                    cands.append((a, v))
                states.append(State(q, cands, masks))
    uniq = {}
    for s in states:
        uniq[s.key()] = s
    return list(uniq.values())


class Planner:
    """Fits the estimators A, P, V, Q, G and the DB-time price models on a training replay table.

    repair_allowed: REPAIR variants available in the table (default: all of REPAIR_MASKS)
    no_opt:         ablation that hides optimizer-derived features from every estimator
    calibrate:      isotonic calibration of V and Q on out-of-fold predictions
    voi_mode:       "diff" regresses the value of information directly; "decomp" factorizes it into
                    P(decision changes) x gain given a change
    """

    def __init__(self, rows, items, seed=0, n_masks=4, log=print, repair_allowed=None, no_opt=False,
                 calibrate=False, voi_mode="diff"):
        self.calibrate, self.voi_mode = calibrate, voi_mode
        self.V_iso = self.Q_iso = None
        self.no_opt = no_opt
        self.repair_allowed = set(repair_allowed) if repair_allowed else set(REPAIR_MASKS)
        self.F = Feat(rows, items, no_opt)
        self.rows = rows
        self.log = log
        qs = [r["question_id"] for r in rows]
        self.qidx = {q: i for i, q in enumerate(qs)}
        n = len(qs)
        gk = list(GroupKFold(5).split(np.zeros(n), groups=np.arange(n)))
        self.fold = np.zeros(n, int)
        for f, (_, te) in enumerate(gk):
            self.fold[te] = f
        self.train_folds = [tr for tr, _ in gk]
        self.rng = random.Random(seed)
        self.n_masks = n_masks
        self.costs()

    def fo(self, q):            # out-of-fold model index for a training question
        return int(self.fold[self.qidx[q]])

    # ---------------------------------------------------------------- price models (train only)
    def costs(self):
        rows = self.rows
        self.llm_cost = {}
        for a in LLM_ACTS:
            vals = []
            for r in rows:
                if a == "REPAIR":
                    vals += [r["REPAIR"][m]["cost"] for m in r["REPAIR"]]
                elif r.get(a):
                    vals.append(r[a]["cost"])
            self.llm_cost[a] = float(np.mean(vals)) if vals else 0.0
        # DB ms predictors from visible pre-probe features (EXPLAIN is mandatory)
        X, yv, ye = [], [], []
        for r in rows:
            for k in ("c0", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN"):
                e = r.get(k)
                if not e:
                    continue
                f = self.price_feats(e)
                X.append(f); yv.append(math.log1p(e["db_ms"].get(VP, 0.0))); ye.append(math.log1p(e["db_ms"].get(EX, 0.0)))
        self.price_names = sorted(set().union(*X))
        M = np.array([[d.get(k, 0) for k in self.price_names] for d in X], float)
        self.pm = {VP: HistGradientBoostingRegressor(max_depth=3, max_iter=150).fit(M, yv),
                   EX: HistGradientBoostingRegressor(max_depth=3, max_iter=150).fit(M, ye)}
        self.mand_ms_mean = float(np.mean([sum(r["c0"]["db_ms"].get(op, 0) for op in MAND) for r in rows]))

    def price_feats(self, e):
        f = dict(sql_features(e["sql"]))
        ex = e["ev"].get("EXPLAIN") or {}
        if ex.get("valid") and "total_cost" in ex and not self.no_opt:
            f["log_cost"] = math.log1p(ex["total_cost"]); f["log_rows"] = math.log1p(ex.get("plan_rows", 0))
        f["full_scans"] = ex.get("full_scans", 0) if ex.get("valid") else 0
        return f

    def pred_ms(self, e, op):
        f = self.price_feats(e)
        x = np.array([[f.get(k, 0) for k in self.price_names]], float)
        return float(max(0.0, math.expm1(self.pm[op].predict(x)[0])))

    # ---------------------------------------------------------------- fitting
    def fit(self, lambdas=LAMBDAS, db_price=1.0):
        F, rows = self.F, self.rows
        self.db_price = db_price
        S = []
        for r in rows:
            S += sample_states(r, self.rng, self.repair_allowed, self.n_masks)
        self.log(f"  states: {len(S)} ({len(S)/len(rows):.0f}/question)")
        groups_s = [self.qidx[s.q] for s in S]
        # A: acceptor
        X, y, g = [], [], []
        for s in S:
            row = F.rows[s.q]
            for i in range(len(s.cands)):
                X.append(F.cand(s, i)); y.append(int(entry(row, s.cands[i])["correct"])); g.append(self.qidx[s.q])
        self.A = Est().fit(X, y, g, self.train_folds)
        self.log(f"  A fitted on {len(X)} candidate-states")
        # P: pairwise preference on decisive cluster pairs (OOF acceptor scores as features)
        av = self.A.predict(X, [int(self.fold[gi]) for gi in g])
        k = 0
        PX, Py, Pg = [], [], []
        for s in S:
            row = F.rows[s.q]
            n = len(s.cands)
            Av = av[k:k + n]; feats = X[k:k + n]; k += n
            cl = defaultdict(list)
            for i in range(n):
                fk = vis_fp(entry(row, s.cands[i]), s.masks[i])
                cl[("fp", fk) if fk else ("u", i)].append(i)
            rp = [max(gr, key=lambda i: Av[i]) for gr in cl.values()]
            for a in rp:
                for b in rp:
                    ca, cb = entry(row, s.cands[a])["correct"], entry(row, s.cands[b])["correct"]
                    if a != b and ca != cb:
                        PX.append(F.pair(feats[a], feats[b], Av[a], Av[b])); Py.append(int(ca)); Pg.append(self.qidx[s.q])
        self.P = Est().fit(PX, Py, Pg, self.train_folds)
        self.log(f"  P fitted on {len(PX)} decisive pairs")
        # V: value of a state under verified selection (OOF selection)
        sel = select_batch(S, F, self.A, self.P, self.fo)
        VX, Vy = [], []
        for s, (b, ab, a2) in zip(S, sel):
            VX.append(F.state(s, b, ab, a2)); Vy.append(int(entry(F.rows[s.q], s.cands[b])["correct"]))
        self.V = Est().fit(VX, Vy, groups_s, self.train_folds)
        self.log(f"  V fitted on {len(VX)} states (selection accuracy {np.mean(Vy):.3f})")
        if self.calibrate:   # isotonic calibration on out-of-fold predictions (Prop. 1 assumes calibration)
            vo = self.V.predict(VX, [self.fo(s.q) for s in S])
            self.V_iso = IsotonicRegression(out_of_bounds="clip").fit(vo, Vy)
            self.log(f"  V calibrated: ECE oof {ece(vo, Vy):.3f} -> {ece(self.V_iso.predict(vo), Vy):.3f}")
        # Q: LLM / rewrite actions, new candidate unobserved
        QS, QA, Qsrc = [], [], []
        for s in S:
            row = F.rows[s.q]
            for a in legal_llm(row, s, self.repair_allowed):
                QS.append(apply_llm(s, a)); QA.append(a); Qsrc.append(s)
        sel2 = select_batch(QS, F, self.A, self.P, self.fo)
        selS = {s.key(): x for s, x in zip(S, sel)}
        QX, Qy, Qg = [], [], []
        for s0, a, s1, (b, _, _) in zip(Qsrc, QA, QS, sel2):
            bb, ab, a2 = selS[s0.key()]
            f = F.state(s0, bb, ab, a2)
            for x in LLM_ACTS:
                f["act_" + x] = int(x == a)
            QX.append(f); Qy.append(int(entry(F.rows[s1.q], s1.cands[b])["correct"])); Qg.append(self.qidx[s0.q])
        self.Q = Est().fit(QX, Qy, Qg, self.train_folds)
        self.log(f"  Q fitted on {len(QX)} state-actions")
        if self.calibrate:
            qo = self.Q.predict(QX, [int(self.fold[gi]) for gi in Qg])
            self.Q_iso = IsotonicRegression(out_of_bounds="clip").fit(qo, Qy)
            self.log(f"  Q calibrated: ECE oof {ece(qo, Qy):.3f} -> {ece(self.Q_iso.predict(qo), Qy):.3f}")
        # G: value of information of evidence actions, targets from OOF V/Q at the realized outcome
        ES, Einfo = [], []
        for s in S:
            for (i, op) in legal_ev(s):
                ES.append(s.add_obs(i, op)); Einfo.append((s, i, op))
        U, D = self.continuation(ES, lambdas, oof=True, with_dec=True)
        U0l, D0l = self.continuation(S, lambdas, oof=True, with_dec=True)
        U0 = {s.key(): u for s, u in zip(S, U0l)}
        D0 = {s.key(): d for s, d in zip(S, D0l)}
        GX, Gy, Gg, Gc = [], [], [], []
        for (s, i, op), u, dec in zip(Einfo, U, D):
            b, ab, a2 = selS[s.key()]
            base = self.ev_feats(s, b, ab, a2, i, op)
            for lam, uv, u0, d1, d0 in zip(lambdas, u, U0[s.key()], dec, D0[s.key()]):
                f = dict(base); f["lam"] = lam
                # value of information: realized continuation value after the observation minus the
                # value of deciding now, both from the same out-of-fold models (biases cancel)
                GX.append(f); Gy.append(uv - u0); Gg.append(self.qidx[s.q]); Gc.append(int(d1 != d0))
        if self.voi_mode == "decomp":
            # Prop. 1: VOI is zero when the decision does not change. Model P(change) and the gain given
            # a change separately, so unchanged-decision noise and rare high-value changes are separated.
            self.G_change = Est().fit(GX, Gc, Gg, self.train_folds)
            idx = [j for j, c in enumerate(Gc) if c]
            self.G_gain = Est("reg").fit([GX[j] for j in idx], [Gy[j] for j in idx], [Gg[j] for j in idx],
                                         self.train_folds)
            self.log(f"  G decomposed: {len(GX)} targets, decision changes in {np.mean(Gc):.3f}")
        else:
            self.G = Est("reg").fit(GX, Gy, Gg, self.train_folds)
            self.log(f"  G fitted on {len(GX)} evidence-action targets")
        self.lambdas = lambdas
        return self

    def ev_feats(self, s, b, ab, a2, i, op, F=None):
        F = F or self.F
        f = F.state(s, b, ab, a2)
        t = F.cand(s, i)
        for k in ("is_vp", "is_exec", "trig", "sql_literals", "sql_joins", "sql_aggs", "ex_log_cost",
                  "res_agree_n", "src_flash", "src_qwen", "src_pro", "src_repair", "src_resample", "src_valuefix"):
            f["tg_" + k] = t.get(k, 0)
        f["tg_is_best"], f["tg_is_c0"] = int(i == b), int(i == 0)
        f["ev_vp"], f["ev_ex"] = int(op == VP), int(op == EX)
        e = entry(F.rows[s.q], s.cands[i])
        f["tg_pred_ms"] = math.log1p(self.pred_ms(e, op))
        return f

    def cal_V(self, v):
        return self.V_iso.predict(np.asarray(v)) if self.V_iso is not None else np.asarray(v)

    def cal_Q(self, q):
        return self.Q_iso.predict(np.asarray(q)) if self.Q_iso is not None else np.asarray(q)

    def voi(self, X, fold_ids):
        if self.voi_mode == "decomp":
            return self.G_change.predict(X, fold_ids) * self.G_gain.predict(X, fold_ids)
        return self.G.predict(X, fold_ids)

    def continuation(self, states, lambdas, oof, with_dec=False):
        """U_l(s) = max(V(s), max_a Q(s,a) - l*C(a)) for every l (batched)."""
        F = self.F
        fo = (lambda q: self.fo(q)) if oof else (lambda q: -1)
        sel = select_batch(states, F, self.A, self.P, fo)
        VX, QX, Qmap = [], [], []
        for si, (s, (b, ab, a2)) in enumerate(zip(states, sel)):
            f = F.state(s, b, ab, a2); VX.append(f)
            for a in legal_llm(F.rows[s.q], s, self.repair_allowed):
                g = dict(f)
                for x in LLM_ACTS:
                    g["act_" + x] = int(x == a)
                QX.append(g); Qmap.append((si, a))
        v = self.cal_V(self.V.predict(VX, [fo(s.q) for s in states]))
        qv = self.cal_Q(self.Q.predict(QX, [fo(states[si].q) for si, _ in Qmap])) if QX else []
        best_q = defaultdict(list)
        for (si, a), x in zip(Qmap, qv):
            best_q[si].append((x, a))
        out, decs = [], []
        for si in range(len(states)):
            row, drow = [], []
            for lam in lambdas:
                u, d = v[si], "ACCEPT"
                for x, a in best_q[si]:
                    if x - lam * self.price_llm(a) > u:
                        u, d = x - lam * self.price_llm(a), a
                row.append(u); drow.append(d)
            out.append(row); decs.append(drow)
        return (out, decs) if with_dec else out

    def price_llm(self, a):
        # an LLM action also pays the mandatory probes on its new candidate
        return self.llm_cost[a] + self.db_price * self.mand_ms_mean / 1000.0


# =============================================================================== online policies
class Runner:
    """Serves questions with a frozen Planner. Only visible evidence is used for decisions; every
    acquired observation is charged its measured DB time."""

    def __init__(self, planner, rows, items, db_price=1.0, oof=False):
        """oof=True serves TRAINING questions with the fold model that did not see them
        (cross-fitted calibration of operating points; the test split is never used)."""
        self.oof = oof
        self.pl = planner
        self.F = Feat(rows, items, planner.no_opt)
        self.rows = {r["question_id"]: r for r in rows}
        self.db_price = db_price
        # memoization: the estimators are frozen, so values depend only on the state (and lambda);
        # different operating points revisit the same states many times
        self._m_sel, self._m_V, self._m_Q, self._m_G, self._m_ms = {}, {}, {}, {}, {}

    def fid(self, q):
        return self.pl.fo(q) if self.oof else -1

    def _sel(self, s):
        k = s.key()
        if k not in self._m_sel:
            self._m_sel[k] = select_batch([s], self.F, self.pl.A, self.pl.P, self.fid)[0]
        return self._m_sel[k]

    def V(self, s, sel):
        k = s.key()
        if k not in self._m_V:
            b, ab, a2 = sel
            self._m_V[k] = float(self.pl.cal_V(self.pl.V.predict([self.F.state(s, b, ab, a2)], [self.fid(s.q)]))[0])
        return self._m_V[k]

    def Qs(self, s, sel):
        k = s.key()
        if k in self._m_Q:
            return self._m_Q[k]
        b, ab, a2 = sel
        row = self.rows[s.q]
        acts = legal_llm(row, s, self.pl.repair_allowed)
        out = {}
        if acts:
            base = self.F.state(s, b, ab, a2)
            X = []
            for a in acts:
                g = dict(base)
                for x in LLM_ACTS:
                    g["act_" + x] = int(x == a)
                X.append(g)
            out = dict(zip(acts, self.pl.cal_Q(self.pl.Q.predict(X, [self.fid(s.q)] * len(X)))))
        self._m_Q[k] = out
        return out

    def Gs(self, s, sel, lam, evs):
        k = (s.key(), lam)
        if k not in self._m_G:
            b, ab, a2 = sel
            allev = legal_ev(s)
            X = []
            for (i, op) in allev:
                f = self.pl.ev_feats(s, b, ab, a2, i, op, F=self.F); f["lam"] = lam; X.append(f)
            self._m_G[k] = dict(zip(allev, self.pl.voi(X, [self.fid(s.q)] * len(X)))) if X else {}
        g = self._m_G[k]
        return {e: g[e] for e in evs if e in g}

    def ev_price(self, s, i, op):
        k = (s.q, s.cands[i], op)
        if k not in self._m_ms:
            self._m_ms[k] = self.pl.pred_ms(entry(self.rows[s.q], s.cands[i]), op)
        return self.db_price * self._m_ms[k] / 1000.0

    def run(self, q, policy, lam=0.0, theta=None, horizon=8, max_db_actions=6, voi_margin=0.0, tau=None):
        row = self.rows[q]
        s = State(q, [("c0", None)], [frozenset()])
        c0 = row["c0"]
        cost = c0["cost"] + self.db_price * sum(c0["db_ms"].get(op, 0) for op in MAND) / 1000.0
        db_ms = sum(c0["db_ms"].get(op, 0) for op in MAND)
        wall = c0["llm_ms"] + db_ms
        trace = []

        def acquire(s, i, op, why, pred=None):
            nonlocal cost, db_ms, wall
            e = entry(row, s.cands[i])
            ms = e["db_ms"].get(op, 0.0)
            cost += self.db_price * ms / 1000.0; db_ms += ms; wall += ms
            trace.append({"action": op, "cand": s.cands[i][0], "pred_cost": pred, "db_ms": round(ms, 2), "why": why})
            return s.add_obs(i, op)

        def generate(s, a, why, score=None):
            nonlocal cost, db_ms, wall
            s2 = apply_llm(s, a)
            e = entry(row, s2.cands[-1])
            mand = sum(e["db_ms"].get(op, 0) for op in MAND)
            cost += e["cost"] + self.db_price * mand / 1000.0; db_ms += mand; wall += e["llm_ms"] + mand
            trace.append({"action": a, "variant": s2.cands[-1][1], "cost": round(e["cost"], 3), "score": score, "why": why})
            return s2

        if policy == "cheap_only":
            pass
        elif policy == "oracle":
            best = None
            for a in [None] + list(LLM_ACTS):
                if a is None:
                    if c0["correct"]:
                        best = ("c0", 0.0); break
                    continue
                variants = [m for m in REPAIR_MASKS if m in self.pl.repair_allowed and m in row["REPAIR"]]
                for v in (variants if a == "REPAIR" else [None]):
                    e = row["REPAIR"][v] if a == "REPAIR" else row.get(a)
                    if e and e["correct"] and (best is None or e["cost"] < best[1]):
                        best = (a, e["cost"])
            ok = best is not None
            return {"q": q, "correct": int(ok), "cost": cost + (best[1] if ok else 0), "db_ms": db_ms,
                    "llm_cost": c0["cost"] + (best[1] if ok else 0),
                    "wall": wall, "trace": [{"action": best[0] if ok else "none"}]}
        else:
            steps = 0
            while steps < horizon:
                steps += 1
                # 1) bundle-style evidence for bundle policies (before deciding)
                if policy in ("fixed_bundle", "optimizer_gated", "rule_selective", "fixed_vp_only", "fixed_ex_only"):
                    for i in range(len(s.cands)):
                        e = entry(row, s.cands[i])
                        for op in OPT:
                            if op in s.masks[i]:
                                continue
                            take = policy == "fixed_bundle" or (policy == "fixed_vp_only" and op == VP) \
                                or (policy == "fixed_ex_only" and op == EX)
                            if policy == "optimizer_gated":
                                if op == VP:
                                    take = True
                                else:
                                    exv = e["ev"].get("EXPLAIN") or {}
                                    c = exv.get("total_cost")
                                    take = theta is None or c is None or c <= theta
                            if policy == "rule_selective":
                                if op == VP:
                                    take = sql_features(e["sql"])["sql_literals"] > 0
                                else:   # execute when results can be compared, or nothing else is known
                                    take = len(s.cands) > 1 or sql_features(e["sql"])["sql_literals"] == 0 \
                                        or (VP in s.masks[i] and not (e["ev"].get(VP) or {}).get("n_missing"))
                            if take:
                                s = acquire(s, i, op, policy)
                sel = self._sel(s)
                v = self.V(s, sel)
                qs = self.Qs(s, sel)
                options = [("ACCEPT", None, v)]
                for a, x in qs.items():
                    options.append(("LLM", a, x - lam * self.pl.price_llm(a)))
                kind, arg, score = max(options, key=lambda o: o[2])
                if policy.startswith("adaptive"):
                    # myopic value-of-information rule: buy the observation whose expected
                    # continuation value exceeds the best decision available now by more than its price
                    banned = {"adaptive_noVP": VP, "adaptive_noEX": EX}.get(policy)   # ablations
                    evs = [(i, op) for (i, op) in legal_ev(s) if op != banned]
                    if sum(len(m) for m in s.masks) >= max_db_actions:
                        evs = []
                    best_ev = None
                    for (i, op), voi in self.Gs(s, sel, lam, evs).items():
                        price = self.ev_price(s, i, op)
                        net = voi - lam * price - voi_margin
                        if net > 0 and (best_ev is None or net > best_ev[1]):
                            best_ev = ((i, op), net)
                    if best_ev is not None:
                        kind, arg, score = "EV", best_ev[0], best_ev[1]
                if kind == "ACCEPT" and tau is not None and v < tau and "SWITCH_QWEN" in qs:
                    # gated escalation (two-phase rule of the original planner): uncertain state -> strong model
                    kind, arg, score = "LLM", "SWITCH_QWEN", v
                if kind == "ACCEPT":
                    trace.append({"action": "ACCEPT", "V": round(v, 3)})
                    break
                if kind == "LLM":
                    s = generate(s, arg, "argmax", round(score, 4))
                else:
                    i, op = arg
                    s = acquire(s, i, op, f"voi={score:.4f}", round(self.ev_price(s, i, op), 5))
        b, _, _ = self._sel(s)
        fin = entry(row, s.cands[b])
        return {"q": q, "correct": int(fin["correct"]), "cost": cost, "db_ms": db_ms, "wall": wall,
                "llm_cost": cost - self.db_price * db_ms / 1000.0,
                "final": s.cands[b][0], "n_llm": len(s.cands), "n_vp": sum(VP in m for m in s.masks),
                "n_ex": sum(EX in m for m in s.masks), "trace": trace}
