"""Operator cost per data scale on the pg_test candidates (1x from the replay table, 10x / 100x measured).

Usage:  python -m experiments.scaling.cost_summary
Output: results/tables/scale_costs.tex (and the numbers on stdout)
"""
import numpy as np

from plangate.jsonl import iter_jsonl
from plangate.paths import REPLAY_DIR, TABLES_DIR

OPS = [("EXPLAIN", "\\op{explain}"), ("PARSE_BIND", "\\op{parse/bind}"), ("VALUE_PROBE", "\\op{value probe}"),
       ("EXEC_BOUNDED", "\\op{bounded exec}")]
CANDIDATES = ("c0", "RESAMPLE_1", "SWITCH_PRO", "SWITCH_QWEN")
SCALES = ("1", "10", "100")


def operator_costs():
    """{scale: {op: [ms per candidate]}} for the first candidate, resample and model switches."""
    rows = {r["question_id"]: r for r in iter_jsonl(REPLAY_DIR / "pg_test.jsonl")}
    res = {"1": {op: [] for op, _ in OPS}}
    for r in rows.values():
        for k in CANDIDATES:
            if r.get(k):
                for op, _ in OPS:
                    res["1"][op].append(r[k]["db_ms"][op])
    for k in SCALES[1:]:
        d = {op: [] for op, _ in OPS}
        for x in iter_jsonl(REPLAY_DIR / f"pg_test.x{k}_costs.jsonl"):
            if x["cand"] in CANDIDATES:
                for op, _ in OPS:
                    d[op].append(x["op_ms"][op])
        res[k] = d
    return res


def main():
    res = operator_costs()
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{1$\times$ (1\,GB)} & \multicolumn{2}{c}{10$\times$ (10\,GB)} & \multicolumn{2}{c}{100$\times$ (101\,GB)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Operator & mean & p95 & mean & p95 & mean & p95 \\", r"\midrule"]
    for op, lab in OPS:
        cells = []
        for k in SCALES:
            a = np.array(res[k][op], float)
            cells += [f"{a.mean():.1f}", f"{np.percentile(a, 95):.0f}"]
            print(k, op, f"mean {a.mean():.1f} p95 {np.percentile(a,95):.0f} max {a.max():.0f} n {len(a)}")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "scale_costs.tex").write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main()
