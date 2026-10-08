"""Simulate realistic traffic against the heart-disease API (through the gcloud proxy).

Happy flow : real patients (raw UCI files + models/reference.csv) with small random jitter -> /predict
Sparse     : real patients with most fields dropped -> 200 + x-low-information header
Negative   : impossible / malformed bodies -> 422
Batch      : /predict_batch with valid batches of several sizes, plus empty, oversized and poisoned batches

Usage:  python scripts/simulate_traffic.py [--base http://localhost:8080] [--seed 42]
"""
import argparse
import collections
import csv
import glob
import json
import random
import time
import urllib.error
import urllib.request

FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
          "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
# (low, high, max jitter) for the continuous fields, matching the API's validation limits
NUMERIC = {"age": (18, 120, 3), "trestbps": (0, 250, 8), "chol": (0, 700, 20),
           "thalach": (50, 230, 8), "oldpeak": (-3, 7, 0.3)}
MISSING = {"", "?", "nan", "NaN", "None"}


def clean(raw: dict) -> dict:
    """Keep the 13 input fields, drop missing values, cast whole numbers to int."""
    body = {}
    for f in FIELDS:
        v = str(raw.get(f, "")).strip()
        if v in MISSING:
            continue
        x = float(v)
        body[f] = round(x, 1) if f == "oldpeak" else int(round(x))
    return body


def load_patients() -> list[dict]:
    rows = []
    for path in sorted(glob.glob("data/raw/*.data")):          # UCI files: no header, 14 columns, '?' = missing
        with open(path) as fh:
            for line in csv.reader(fh):
                if len(line) >= 13:
                    rows.append(clean(dict(zip(FIELDS, line))))
    try:
        with open("models/reference.csv") as fh:                # training-distribution sample, has a header
            rows += [clean(r) for r in csv.DictReader(fh)]
    except FileNotFoundError:
        pass
    rows = [r for r in rows if "age" in r and "sex" in r and 18 <= r["age"] <= 120]
    if not rows:
        raise SystemExit("no patients found: run from the repo root after dvc pull")
    return rows


def jitter(p: dict, rng: random.Random) -> dict:
    """Move each continuous value a little up or down, then keep it inside the API's rules."""
    q = dict(p)
    for f, (lo, hi, step) in NUMERIC.items():
        if f in q:
            if f == "oldpeak" and q[f] == 0:
                continue                                        # ~40% of patients sit exactly at 0: keep that spike intact
            v = q[f] + rng.uniform(-step, step)
            if f == "oldpeak" and q[f] > 0:
                v = max(v, 0.1)                                 # a positive reading stays positive
            v = min(max(v, lo), hi)
            q[f] = round(v, 1) if f == "oldpeak" else int(round(v))
    if "thalach" in q:                                          # cross-field rule: thalach <= 260 - age
        q["thalach"] = max(50, min(q["thalach"], 260 - q["age"]))
    return q


def sparse(p: dict) -> dict:
    """Only age and sex: valid, but too little clinical information."""
    return {"age": p["age"], "sex": p["sex"]}


def negatives() -> list[tuple[str, object]]:
    base = {"age": 55, "sex": 1, "cp": 4, "trestbps": 140, "chol": 250, "thalach": 150}
    return [
        ("age too low", {**base, "age": 5}),
        ("age too high", {**base, "age": 150}),
        ("sex not 0/1", {**base, "sex": 2}),
        ("cp not in 1-4", {**base, "cp": 9}),
        ("chol too high", {**base, "chol": 950}),
        ("trestbps negative", {**base, "trestbps": -10}),
        ("thal not 3/6/7", {**base, "thal": 5}),
        ("oldpeak too high", {**base, "oldpeak": 12}),
        ("thalach > 260-age", {**base, "age": 80, "thalach": 200}),
        ("unknown field", {**base, "bmi": 27}),
        ("age is text", {**base, "age": "fifty"}),
        ("missing age", {k: v for k, v in base.items() if k != "age"}),
        ("empty body", {}),
        ("not an object", [1, 2, 3]),
    ]


def post(base: str, path: str, body) -> tuple[int, dict, float, object]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read() or b"null")
            return r.status, dict(r.headers), (time.perf_counter() - t0) * 1000, out
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), (time.perf_counter() - t0) * 1000, e.read()[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--happy", type=int, default=350)
    ap.add_argument("--sparse", type=int, default=15)
    ap.add_argument("--neg-rounds", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    patients = load_patients()
    print(f"loaded {len(patients)} real patients")

    report = collections.OrderedDict()
    latencies = []
    surprises = []

    def check(name, expected, code, extra_ok=True, detail=""):
        r = report.setdefault(name, collections.Counter())
        r["sent"] += 1
        ok = code == expected and extra_ok
        r["ok" if ok else "unexpected"] += 1
        if not ok and len(surprises) < 15:
            surprises.append(f"{name}: got {code}, expected {expected} {detail}")

    # 1. happy flow
    for p in rng.choices(patients, k=args.happy):
        code, _, ms, body = post(args.base, "/predict", jitter(p, rng))
        latencies.append(ms)
        check("happy /predict (expect 200)", 200, code, detail=str(body)[:120])

    # 2. sparse patients: accepted but flagged
    for p in rng.choices(patients, k=args.sparse):
        code, hdr, ms, _ = post(args.base, "/predict", sparse(p))
        latencies.append(ms)
        flagged = any(k.lower() == "x-low-information" for k in hdr)
        check("sparse /predict (expect 200 + x-low-information)", 200, code, flagged, "header missing" if not flagged else "")

    # 3. negative flow
    for _ in range(args.neg_rounds):
        for label, body in negatives():
            code, _, _, _ = post(args.base, "/predict", body)
            check(f"negative: {label} (expect 422)", 422, code)

    # 4. batch endpoint
    batch_rows = 0
    for size in (1, 10, 25, 50, 100):
        batch = [jitter(p, rng) for p in rng.choices(patients, k=size)]
        code, _, ms, body = post(args.base, "/predict_batch", batch)
        n_out = len(body) if isinstance(body, list) else -1
        batch_rows += size if code == 200 else 0
        check("batch valid (expect 200, one result per patient)", 200, code, n_out == size, f"size {size}, got {n_out} results")
        print(f"  batch of {size:>3}: {code} in {ms:.0f} ms, {n_out} results")
    bad = [jitter(p, rng) for p in rng.choices(patients, k=9)] + [{"age": 5, "sex": 1}]
    for label, body in [("empty list", []),
                        ("501 patients", [jitter(p, rng) for p in rng.choices(patients, k=501)]),
                        ("one bad patient among 10", bad)]:
        code, _, _, _ = post(args.base, "/predict_batch", body)
        check(f"batch negative: {label} (expect 422)", 422, code)

    # report
    print()
    total = 0
    for name, c in report.items():
        total += c["sent"]
        flag = "" if c["unexpected"] == 0 else f"   <-- {c['unexpected']} unexpected"
        print(f"{c['sent']:>4} sent  {c['ok']:>4} ok  {name}{flag}")
    stored = report["happy /predict (expect 200)"]["ok"] + report["sparse /predict (expect 200 + x-low-information)"]["ok"] + batch_rows
    latencies.sort()
    print(f"\n{total} requests, about {stored} prediction rows written to BigQuery")
    if latencies:
        print(f"client-side latency /predict: p50 {latencies[len(latencies)//2]:.0f} ms, "
              f"p95 {latencies[int(len(latencies)*0.95)]:.0f} ms (includes the proxy hop)")
    if surprises:
        print("\nfirst unexpected results:")
        print("\n".join("  " + s for s in surprises))


if __name__ == "__main__":
    main()
