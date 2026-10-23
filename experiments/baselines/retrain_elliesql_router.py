"""In-domain re-training of the EllieSQL router (adapts the router to the backbone model used here).

The released router was trained on pipeline outcomes of gpt-4o-mini. Here it is adapted to the
backbone actually used (deepseek-v4-flash): labels = cheapest pipeline that is correct on the
cached EllieSQL runs (waterfall labelling of the paper; UNSOLVED -> ADVANCED as in
src/router/labeler.py). Training starts from the released DPO checkpoint and fine-tunes the top
layers + classification head, cross-fitted with the same 5 folds as experiments/crossfit/planner.py.

Outputs results/baselines/elliesql_router_indomain.json and merges EllieSQL policies with the adapted
router (argmax, and a threshold sweep on P(basic)) into results/baselines/published_methods_sqlite.json.
Usage: python -m experiments.baselines.retrain_elliesql_router
"""
from __future__ import annotations
import json, time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from plangate.baselines import elliesql as E
from plangate.benchmark import load_dev
from plangate.jsonl import iter_jsonl
from plangate.paths import BASELINES_DIR

REPO, SUB = "derrickzhu/EllieSQL_Router_Checkpoints", "qwen_dpo"
LEVELS = ["G_B", "G_M", "G_A"]
EPOCHS, LR, BATCH, TRAIN_TOP_LAYERS, MAXLEN = 3, 2e-5, 4, 4, 160   # router inputs are <= 130 tokens


def label_of(r):
    for i, l in enumerate(LEVELS):
        if r[l]["correct"]:
            return i
    return 2   # UNSOLVED -> ADVANCED


def load_model():
    tok = AutoTokenizer.from_pretrained(REPO, subfolder=SUB)
    m = AutoModelForSequenceClassification.from_pretrained(REPO, subfolder=SUB)
    for p in m.parameters():
        p.requires_grad = False
    layers = m.model.layers
    for layer in layers[-TRAIN_TOP_LAYERS:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in list(m.model.norm.parameters()) + list(m.score.parameters()):
        p.requires_grad = True
    return tok, m


def encode(tok, texts):
    return tok(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLEN)


def main():
    torch.manual_seed(42); np.random.seed(42)
    items = {x["question_id"]: x for x in load_dev()}
    rows = sorted(iter_jsonl(BASELINES_DIR / "elliesql.jsonl"), key=lambda r: r["question_id"])
    texts = [E.router_input(items[r["question_id"]]["question"], r["keys_only"]) for r in rows]
    y = np.array([label_of(r) for r in rows])
    n = len(rows)
    tok, _ = load_model()
    lens = [len(tok(t)["input_ids"]) for t in texts]
    print(f"{n} questions; label dist {np.bincount(y, minlength=3).tolist()}; "
          f"router input tokens median {int(np.median(lens))}, max {max(lens)} (truncated at {MAXLEN})", flush=True)

    probs = np.zeros((n, 3))
    folds = list(GroupKFold(5).split(np.zeros(n), groups=np.arange(n)))   # same split as the cross-fitted planner
    t0 = time.time()
    for f, (tr, te) in enumerate(folds):
        tok, m = load_model()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
        # class-balanced loss (labels are skewed towards BASIC)
        w = torch.tensor(len(tr) / (3 * np.maximum(np.bincount(y[tr], minlength=3), 1)), dtype=torch.float32)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        m.train()
        for ep in range(EPOCHS):
            perm = np.random.permutation(tr)
            tot = 0.0
            for b in range(0, len(perm), BATCH):
                idx = perm[b:b + BATCH]
                enc = encode(tok, [texts[i] for i in idx])
                loss = lossf(m(**enc).logits, torch.tensor(y[idx]))
                loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item() * len(idx)
            print(f"  fold {f} epoch {ep} loss {tot/len(perm):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        m.eval()
        with torch.no_grad():
            for b in range(0, len(te), 8):
                idx = te[b:b + 8]
                probs[idx] = torch.softmax(m(**encode(tok, [texts[i] for i in idx])).logits, -1).numpy()
        del m

    pred = probs.argmax(1)
    released = np.array([LEVELS.index(r["route"]) for r in rows])
    basic_fails = np.array([int(not r["G_B"]["correct"]) for r in rows])
    rel_p = np.array([r["route_probs"][1] + r["route_probs"][2] for r in rows])
    print(f"\nlabel accuracy: released router {np.mean(released == y):.3f} | in-domain router {np.mean(pred == y):.3f}")
    print(f"AUROC P(not basic) for 'basic fails': released {roc_auc_score(basic_fails, rel_p):.3f} | "
          f"in-domain {roc_auc_score(basic_fails, probs[:, 1] + probs[:, 2]):.3f}")

    def outcome(r, lvl):
        link, p = r["link"], r[lvl]
        return {"correct": int(p["correct"]), "cost": link["cost"] + p["cost"] + p["db_ms"] / 1000,
                "lat": link["ms"] + p["ms"] + p["db_ms"] + r.get("router_ms", 0.0), "n_llm": link["calls"] + p["calls"]}

    pol = {"EllieSQL (in-domain router)": [outcome(r, LEVELS[k]) for r, k in zip(rows, pred)]}
    for t in (0.3, 0.5, 0.7, 0.9):   # cost-aware routing: BASIC iff P(basic) >= t, else best of G_M / G_A
        ch = [0 if probs[i, 0] >= t else (1 if probs[i, 1] >= probs[i, 2] else 2) for i in range(n)]
        pol[f"EllieSQL (in-domain router, t={t})"] = [outcome(r, LEVELS[k]) for r, k in zip(rows, ch)]
    print(f"\n  {'policy':38s} {'EX':>6} {'cost':>6} {'p95 s':>6}")
    for k, v in pol.items():
        print(f"  {k:38s} {np.mean([x['correct'] for x in v]):6.3f} {np.mean([x['cost'] for x in v]):6.2f} "
              f"{np.percentile([x['lat'] for x in v], 95)/1000:6.1f}")

    (BASELINES_DIR / "elliesql_router_indomain.json").write_text(json.dumps(
        {"question_ids": [r["question_id"] for r in rows], "probs": probs.tolist(), "labels": y.tolist()}), encoding="utf8")
    pb = BASELINES_DIR / "published_methods_sqlite.json"
    d = json.loads(pb.read_text(encoding="utf8"))
    order = {r["question_id"]: i for i, r in enumerate(rows)}
    for k, v in pol.items():
        d["policies"][k] = [v[order[q]] for q in d["question_ids"]]
    pb.write_text(json.dumps(d, default=float), encoding="utf8")
    print("appended to", pb)


if __name__ == "__main__":
    main()
