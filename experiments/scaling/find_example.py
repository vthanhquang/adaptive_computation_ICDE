"""Find a real test question (100x costs, 1.5x budget) where the adaptive policy skips an expensive
execution that the fixed bundle pays for, and both answer correctly. Prints its full traces with the
predicted cost and VOI of each considered observation (re-computed with the frozen planner).

Usage:  python -m experiments.scaling.find_example   (needs results/scaling/pg_x100/planner.pkl)
Output: results/scaling/example.json
"""
import json, pickle

from plangate.evaluation.heldout import chosen_traces, load, run_dir, run_summary
from plangate.paths import SCALING_DIR
from plangate.planning import adaptive as AD

d = run_dir("pg_x100")
tr = {}
for x in chosen_traces("pg_x100"):
    if x["config"] in ("p1.0_B1.5_adaptive", "p1.0_B1.5_optimizer_gated", "p1.0_B2.0_fixed_bundle"):
        tr.setdefault(x["config"], {})[x["q"]] = x
rows, items, _ = load("pg_test", 100)
R = {r["question_id"]: r for r in rows}
best = []
for q, a in tr["p1.0_B1.5_adaptive"].items():
    f = tr["p1.0_B2.0_fixed_bundle"].get(q)
    g = tr["p1.0_B1.5_optimizer_gated"].get(q)
    if not f or not g:
        continue
    if a["correct"] and f["correct"] and a["n_ex"] == 0 and f["db_ms"] - a["db_ms"] > 2000:
        best.append((f["db_ms"] - a["db_ms"], q))
best.sort(reverse=True)
print("candidates:", best[:8])
q = best[0][1]
pl = pickle.loads((d / "planner.pkl").read_bytes())
Rn = AD.Runner(pl, rows, items)
s = AD.State(q, [("c0", None)], [frozenset()])
sel = Rn._sel(s)
lam = run_summary("pg_x100")["chosen"]["1.0"]["1.5"]["adaptive"]["params"]
print("adaptive params", lam)
voi = Rn.Gs(s, sel, lam["lam"], AD.legal_ev(s))
out = {"q": q, "question": items[q]["question"], "sql_c0": R[q]["c0"]["sql"],
       "V_s0": Rn.V(s, sel), "params": lam, "options": []}
for (i, op), g in voi.items():
    e = AD.entry(R[q], s.cands[i])
    out["options"].append({"op": op, "voi": float(g), "pred_ms": Rn.pl.pred_ms(e, op), "actual_ms": e["db_ms"][op],
                           "price_units": Rn.ev_price(s, i, op) * lam["lam"]})
out["explain_cost"] = (R[q]["c0"]["ev"].get("EXPLAIN") or {}).get("total_cost")
out["adaptive_trace"] = tr["p1.0_B1.5_adaptive"][q]["trace"]
out["adaptive_db_ms"] = tr["p1.0_B1.5_adaptive"][q]["db_ms"]
out["bundle_db_ms"] = tr["p1.0_B2.0_fixed_bundle"][q]["db_ms"]
out["gated_db_ms"] = tr["p1.0_B1.5_optimizer_gated"][q]["db_ms"]
out["correct"] = True
print(json.dumps(out, indent=1, default=str))
(SCALING_DIR / "example.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf8")
