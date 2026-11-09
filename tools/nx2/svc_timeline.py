"""Rebuild each frame's critical chain of guest waits and wakes from EDEN_NX2_SVC_TIMELINE.

    python -I tools/nx2/svc_timeline.py <svc-timeline.csv> [--frames N] [--min-wait-us 30]

Walks back from each present (the SendSyncRequest during which the present count changed): the
presenting thread's run before it, the wait that run started from, the SVC that woke that wait
(SignalToAddress, SignalProcessWideKey, ArbitrateUnlock, SignalEvent) and the waker's own run,
and so on. A wait nothing in the game woke (an event signalled by a service or the GPU, a sleep
or a timeout) ends the chain. Prints each step and, over all frames, where the chain's time went.
"""
import argparse
import bisect
import collections
import csv

WAITS = {0x34: "WaitForAddress", 0x1c: "WaitProcessWideKeyAtomic", 0x1a: "ArbitrateLock",
         0x18: "WaitSynchronization", 0x0b: "SleepThread", 0x21: "SendSyncRequest"}
SIGNALS = {0x35: "SignalToAddress", 0x1d: "SignalProcessWideKey", 0x1b: "ArbitrateUnlock",
           0x11: "SignalEvent"}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(in_ns=int(r["in_ns"]), out_ns=int(r["out_ns"]), thread=int(r["thread"]),
                             core=int(r["core"]), svc=int(r["svc"], 16), arg=int(r["arg"], 16),
                             arg2=int(r["arg2"], 16), present=int(r["present"])))
    rows.sort(key=lambda r: r["in_ns"])
    return rows


def wake_key(e):
    """What signal ends wait event e: (signal svc, key) or None when only the outside can."""
    if e["svc"] == 0x34:
        return (0x35, e["arg"])
    if e["svc"] == 0x1c:
        return (0x1d, e["arg2"])
    if e["svc"] == 0x1a:
        return (0x1b, e["arg2"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--min-wait-us", type=float, default=30)
    ap.add_argument("--sleep-is-run", action="store_true",
                    help="count SleepThread and SendSyncRequest as part of a run (polling loops)")
    a = ap.parse_args()
    rows = load(a.csv)
    t0 = rows[0]["in_ns"]
    by_thread = collections.defaultdict(list)
    for r in rows:
        by_thread[r["thread"]].append(r)
    signals = collections.defaultdict(list)  # (svc, key) -> [in_ns]
    for r in rows:
        if r["svc"] in (0x35, 0x1b):
            signals[(r["svc"], r["arg"])].append(r)
        elif r["svc"] == 0x1d:
            signals[(0x1d, r["arg"])].append(r)
    for v in signals.values():
        v.sort(key=lambda r: r["in_ns"])

    # Presents: the SendSyncRequest spanning the change of the present count.
    presents = []
    for p in sorted({r["present"] for r in rows})[1:]:
        before = max(r["in_ns"] for r in rows if r["present"] == p - 1)
        after = min(r["in_ns"] for r in rows if r["present"] == p)
        cands = [r for r in rows if r["svc"] == 0x21 and r["in_ns"] <= after and r["out_ns"] >= before]
        if cands:
            presents.append(min(cands, key=lambda r: r["out_ns"] - r["in_ns"]))
    min_wait = a.min_wait_us * 1000
    totals = collections.Counter()
    frame_spans = []
    for n, pres in enumerate(presents[: a.frames]):
        print(f"\n=== present at {(pres['in_ns'] - t0) / 1e6:.2f} ms by t{pres['thread']}")
        thread, t = pres["thread"], pres["in_ns"]
        chain_start = t
        steps = 0
        while steps < 60:
            steps += 1
            evs = by_thread[thread]
            i = bisect.bisect_right([e["out_ns"] for e in evs], t) - 1
            # The last wait (long enough) that ended at or before t on this thread.
            while i >= 0 and not (evs[i]["svc"] in WAITS and evs[i]["out_ns"] - evs[i]["in_ns"] >= min_wait
                                   and not (a.sleep_is_run and evs[i]["svc"] in (0x0b, 0x21))):
                i -= 1
            if i < 0:
                print(f"  t{thread} runs from before the capture")
                break
            w = evs[i]
            run = t - w["out_ns"]
            totals[f"run t{thread}"] += run
            name = WAITS.get(w["svc"], hex(w["svc"]))
            key = wake_key(w)
            waker = None
            if key is not None:
                cands = signals.get(key, [])
                j = bisect.bisect_right([s["in_ns"] for s in cands], w["out_ns"]) - 1
                if j >= 0 and cands[j]["in_ns"] >= w["in_ns"] - 1000:
                    waker = cands[j]
            wait_ns = w["out_ns"] - w["in_ns"]
            if waker is None:
                totals[f"outside: {name} {w['arg']:#x}"] += wait_ns
                print(f"  t{thread} ran {run / 1e3:7.1f} us after {name}({w['arg']:#x}) "
                      f"{wait_ns / 1e3:.1f} us, woken from outside the game")
                chain_start = w["in_ns"]
                break
            latency = w["out_ns"] - waker["in_ns"]
            totals["wake latency"] += max(latency, 0)
            print(f"  t{thread} ran {run / 1e3:7.1f} us after {name}({w['arg']:#x}) "
                  f"woken by t{waker['thread']} {SIGNALS[waker['svc']]} (+{latency / 1e3:.1f} us)")
            thread, t = waker["thread"], waker["in_ns"]
            chain_start = t
        span = pres["in_ns"] - chain_start
        frame_spans.append(span)
        print(f"  chain covers {span / 1e6:.2f} ms before the present")
    print("\n=== where the chains' time went (ms per frame)")
    nframes = max(1, min(a.frames, len(presents)))
    for k, v in totals.most_common(20):
        print(f"  {v / 1e6 / nframes:7.2f}  {k}")


if __name__ == "__main__":
    main()
