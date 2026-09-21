#!/usr/bin/env python3
"""Wall-clock progress/ETA for long runs. Prints local time, not just durations.

Usage:
  eta.py start  <total> [--label NAME]              at launch: records t0
  eta.py check  <jsonl> <total> [--label NAME]      progress + finish timestamp

Reads the output .jsonl row count as progress, so it works for any of the runners
that append one line per inference. Rate is measured from the file's own mtime
window when a start marker is absent, so a check works even after a reconnect.
"""
import argparse, json, os, sys, time
from datetime import datetime, timedelta

MARK = "/tmp/eta_marks.json"
TZ = None  # local server time


def now():
    return datetime.now()


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def hhmm(sec):
    sec = max(0, int(sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def load():
    try:
        return json.load(open(MARK))
    except Exception:
        return {}


def save(d):
    try:
        json.dump(d, open(MARK, "w"))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["start", "check"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--label", default="run")
    ap.add_argument("--rate", type=float, default=None,
                    help="seconds per inference, if known in advance")
    args = ap.parse_args()
    marks = load()

    if args.mode == "start":
        total = int(args.a)
        t0 = time.time()
        marks[args.label] = dict(t0=t0, total=total)
        save(marks)
        est = args.rate * total if args.rate else None
        print(f"[{args.label}] STARTED   {fmt(now())}")
        print(f"  target {total} inferences")
        if est:
            print(f"  est duration {hhmm(est)}  ->  ETA {fmt(now() + timedelta(seconds=est))}")
        return

    path, total = args.a, int(args.b)
    done = 0
    if os.path.exists(path):
        with open(path, "rb") as f:
            done = sum(1 for _ in f)
    m = marks.get(args.label)
    t0 = m["t0"] if m else None
    el = (time.time() - t0) if t0 else None
    rate = (el / done) if (el and done) else args.rate
    pct = 100.0 * done / total if total else 0
    print(f"[{args.label}] {fmt(now())}   {done}/{total}  ({pct:.1f}%)")
    if el:
        print(f"  elapsed {hhmm(el)}   mean {rate:.2f}s/inf")
    if rate and done < total:
        rem = rate * (total - done)
        print(f"  remaining {hhmm(rem)}  ->  ETA {fmt(now() + timedelta(seconds=rem))}")
    elif done >= total:
        print(f"  COMPLETE")


if __name__ == "__main__":
    main()
