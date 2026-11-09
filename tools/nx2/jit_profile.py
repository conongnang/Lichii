"""Guest-code profile: samples of generated code named by guest block.

    python -I tools/nx2/jit_profile.py <samples-*.txt.raw> <jit map> [--threads CPUCore] [--top 40]

The raw file comes from the debug server's `sample` (or EDEN_NX2_SAMPLE); the map from running
with EDEN_NX2_JIT_MAP=<file> (dynarmic's perf map: "start size name" per compiled block and
thunk). A sample whose top frame is generated code counts for that block; one in host code
called from generated code counts for the host function and for the block that called it.
"""
import argparse
import bisect
import collections


def load_map(path):
    starts, ends, names = [], [], []
    entries = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        start, size = int(parts[0], 16), int(parts[1], 16)
        entries.append((start, start + size, parts[2].strip()))
    # Later registrations win (code space is reused after a cache clear).
    entries.sort(key=lambda e: e[0])
    for s, e, n in entries:
        starts.append(s)
        ends.append(e)
        names.append(n)
    return starts, ends, names


def name_of(addr, m):
    starts, ends, names = m
    i = bisect.bisect_right(starts, addr) - 1
    while i >= 0 and starts[i] <= addr:
        if addr < ends[i]:
            return names[i]
        i -= 1
        if addr - starts[i + 1] > 1 << 20:
            break
    return "jit:unmapped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("map")
    ap.add_argument("--threads", default="CPUCore")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    m = load_map(a.map)
    total = 0
    in_jit = 0
    blocks = collections.Counter()
    host_from_jit = collections.Counter()
    caller_block = collections.Counter()
    for line in open(a.raw, encoding="utf-8", errors="replace"):
        stack, _, count = line.rstrip().rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        frames = stack.split(";")
        if not frames[0].startswith(a.threads):
            continue
        total += n
        leaf = frames[-1]
        if leaf.startswith("jit+"):
            in_jit += n
            blocks[name_of(int(leaf[4:], 16), m)] += n
            continue
        # Host code: find the nearest generated-code caller.
        for f in reversed(frames[1:-1]):
            if f.startswith("jit+"):
                host_from_jit[leaf] += n
                caller_block[name_of(int(f[4:], 16), m)] += n
                break
    print(f"{total} samples on {a.threads}*: {in_jit * 100 / max(total, 1):.1f}% in generated code, "
          f"{sum(host_from_jit.values()) * 100 / max(total, 1):.1f}% in host code called from it")
    print("\n-- generated code by block (% of all samples)")
    for name, n in blocks.most_common(a.top):
        print(f"  {n * 100 / total:6.2f}%  {name}")
    print("\n-- host code called from generated code")
    for name, n in host_from_jit.most_common(20):
        print(f"  {n * 100 / total:6.2f}%  {name}")
    print("\n-- blocks that called into host code")
    for name, n in caller_block.most_common(15):
        print(f"  {n * 100 / total:6.2f}%  {name}")


if __name__ == "__main__":
    main()
