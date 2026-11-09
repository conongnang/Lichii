# OoT performance loop: launch, load the autosave, measure gameplay fps, optionally profile.
#
#   perf.py run <name> [--env K=V ...] [--profile] [--sample S] [--threads GPU,GPUBack,CPUCore_*]
#               [--grep TEXT ...] [--measure N] [--settle S] [--exe lichii.exe] [--attach] [--kill]
#   perf.py results            : the results table
#
# Every switch goes into the launched process's own environment (EDEN_NX2_GPU_PROFILE is read at
# static init, before nx2_switches.txt is parsed). Output: $OOT_PERF_DIR/<name>/ and results.tsv.
# Run with native Windows Python (-I).
import argparse, datetime, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MCP = os.path.join(REPO, "tools", "nx2", "debug_mcp.py")
OUT = os.environ.get("OOT_PERF_DIR", os.path.join(os.environ["TEMP"], "oot-perf"))
EMU_DIR = os.environ.get("LICHII_DIR", "")  # folder with lichii.exe
ROM = os.environ.get("OOT_ROM", "")  # the game file
REFERENCE = os.environ.get("OOT_REFERENCE", "")  # optional screenshot of the benchmark spot
SERVER = "127.0.0.1:6545"
SWITCHES = os.path.join(os.environ["APPDATA"], "eden", "nx2_switches.txt")
def call(method, params=None, out=None, timeout=None):
    cmd = [sys.executable, "-I", MCP, "--server", SERVER,
           "--call", json.dumps({"method": method, "params": params or {}})]
    if out:
        cmd += ["--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"{method}: {r.stdout.strip()} {r.stderr.strip()}")
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"{method}: {data['error']}")
    return data.get("result", data) if isinstance(data, dict) else data


def running():
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq lichii.exe", "/NH"], capture_output=True, text=True)
    return "lichii.exe" in r.stdout.lower()


def launch(env, exe, kill, affinity=None):
    if running():
        if not kill:
            sys.exit("lichii.exe is already running: pass --kill to replace it or --attach to use it")
        subprocess.run(["taskkill", "/F", "/IM", "lichii.exe"], capture_output=True)
        for _ in range(20):
            if not running():
                break
            time.sleep(1)
    with open(SWITCHES, "w", encoding="utf-8") as f:
        f.write(f"# Written by perf.py; switches go in the process environment\nEDEN_NX2_DEBUG_SERVER={SERVER}\n")
    sets = "".join(f"$env:{k}='{v}'; " for k, v in env.items())
    ps = (f"{sets}$p = Start-Process -PassThru -FilePath '{os.path.join(EMU_DIR, exe)}' "
          f"-ArgumentList '-g','\"{ROM}\"' -WorkingDirectory '{EMU_DIR}'")
    if affinity:
        ps += f"; $p.ProcessorAffinity = {int(affinity, 16)}"
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)
    for _ in range(90):
        try:
            if call("status", {"fps_window_ms": 0}, timeout=10)["presents"] >= 10:
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit("the game did not start presenting within 3 minutes")


def press_a():
    call("input_sequence", {"steps": [{"buttons": ["A"], "frames": 4}, {"frames": 2}], "timeout_s": 30},
         timeout=40)


def read_png(path):
    """(width, height, channels, rows) of an 8-bit RGB or RGBA PNG, stdlib only."""
    import struct, zlib
    data = open(path, "rb").read()
    pos, idat, w = 8, b"", 0
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", chunk[:10])
            ch = {2: 3, 6: 4}[color]
        elif kind == b"IDAT":
            idat += chunk
        pos += 12 + length
    raw, stride, rows, prev = zlib.decompress(idat), w * ch, [], bytearray(w * ch)
    for y in range(h):
        f, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - ch] if i >= ch else 0
            b, c = prev[i], prev[i - ch] if i >= ch else 0
            if f == 1: line[i] = (line[i] + a) & 255
            elif f == 2: line[i] = (line[i] + b) & 255
            elif f == 3: line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return w, h, ch, rows


def screen():
    path = os.path.join(OUT, "_probe.png")
    for attempt in range(12):  # a cold shader cache can stall presents for many seconds
        try:
            call("screenshot", out=path, timeout=30)
            break
        except RuntimeError as e:
            if "no frame was presented" not in str(e) or attempt == 11:
                raise
            time.sleep(2)
    w, h, ch, rows = read_png(path)
    return lambda x, y: tuple(rows[y * h // 720][(x * w // 1280) * ch:(x * w // 1280) * ch + 3])


def near(c, ref, tol=22):
    return all(abs(a - b) <= tol for a, b in zip(c, ref))


def in_gameplay(px=None):
    """The HUD's hearts (top left, red) are on screen."""
    px = px or screen()
    red = sum(1 for x in range(56, 132, 2) for y in range(40, 60, 2)
              if (lambda r, g, b: r > 170 and g < 90 and b < 80)(*px(x, y)))
    return red > 25


def on_save_list(px):
    """The Load screen's brown header bar."""
    return all(near(px(x, 30), (89, 69, 38)) for x in (300, 980)) and near(px(640, 30), (114, 99, 76))


def scene_difference(px):
    """Mean difference (0-255) from the OOT_REFERENCE screenshot."""
    w, h, ch, rows = read_png(REFERENCE)
    ref = lambda x, y: tuple(rows[y * h // 720][(x * w // 1280) * ch:(x * w // 1280) * ch + 3])
    total = count = 0
    for y in range(100, 700, 40):
        for x in range(160, 1120, 40):
            total += sum(abs(a - b) for a, b in zip(px(x, y), ref(x, y))) / 3
            count += 1
    return total / count


def press(button, wait=1.2):
    call("input_sequence", {"steps": [{"buttons": [button], "frames": 4}, {"frames": 2}],
                            "timeout_s": 30}, timeout=40)
    time.sleep(wait)


def load_save(slot):
    """A from boot skips the intro and picks Continue; on the Load list, Down `slot` times
    (0 = newest save), A, A on Load. No button is pressed once loading starts: a stray A in
    gameplay walks Link into his house, and the game autosaves there."""
    for _ in range(90):
        px = screen()
        if on_save_list(px):
            break
        if in_gameplay(px):
            print("booted into gameplay (the game resumed its last spot); measuring there")
            return
        press("A", 1.0)
    else:
        sys.exit("never reached the save list")
    time.sleep(1.0)
    # slot -1: the bottom of the list (Down stops at the last save; the list scrolls, so only the
    # scene check after loading can tell it was the right one).
    for _ in range(slot if slot >= 0 else 24):
        press("Down", 0.6 if slot >= 0 else 0.35)
    if slot >= 0 and not near(screen()(285, 110 + 145 * slot), (255, 255, 255), 40):
        sys.exit(f"save slot {slot} is not selected on the Load list")
    press("A", 1.5)
    press("A", 0.5)
    for _ in range(90):
        time.sleep(1)
        px = screen()
        if in_gameplay(px):
            time.sleep(4)
            if not REFERENCE:
                return 0.0
            diff = scene_difference(screen())
            if diff > 35:
                print(f"WARNING: scene differs from OOT_REFERENCE (mean diff {diff:.0f}); "
                      "wrong save slot?", flush=True)
            return diff
    sys.exit("the save did not load")


# A route that shows new things to draw: walk ahead, turn the camera around, walk back, then a full
# camera sweep. In frames (at 60 fps about 18 s).
WALK_ROUTE = [
    {"left_stick": [0, 1], "frames": 240},
    {"right_stick": [1, 0], "frames": 90},
    {"left_stick": [0, 1], "frames": 240},
    {"right_stick": [-1, 0], "frames": 180},
    {"left_stick": [1, 0.3], "frames": 180},
    {"right_stick": [1, 0.2], "frames": 180},
]


def walk(times):
    """Plays WALK_ROUTE `times` times, then reads the presents of that whole span (frame_times
    `past`; the debug server answers one request at a time). Hitches: intervals over 50 ms."""
    started = time.time()
    for _ in range(times):
        call("input_sequence", {"steps": WALK_ROUTE, "timeout_s": 600}, timeout=660)
    seconds = time.time() - started
    ft = call("frame_times", {"past": seconds}, timeout=60)
    return {"seconds": round(seconds, 1), "frames": ft["frames"], "fps": round(ft["fps"], 1),
            "hitches": ft["hitches"], "hitch_ms": ft["hitch_ms"], "max_ms": round(ft["max_ms"], 1),
            "worst": ft["worst_ms"], "truncated": ft["truncated"]}


def git_rev():
    r = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "-uno"], capture_output=True, text=True)
    return r.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def run(a):
    if not a.attach and (not EMU_DIR or not ROM):
        sys.exit("set LICHII_DIR (the folder with lichii.exe) and OOT_ROM (the game file)")
    env = {"EDEN_NX2_DEBUG_SERVER": SERVER}
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    if a.profile:
        env["EDEN_NX2_GPU_PROFILE"] = "1"
    d = os.path.join(OUT, a.name)
    os.makedirs(d, exist_ok=True)
    summary_scene = None
    if not a.attach:
        launch(env, a.exe, a.kill, a.affinity)
        summary_scene = load_save(a.slot)
    time.sleep(a.settle)
    walk_stats = walk(a.walk) if a.walk else None
    samples = [round(call("status", {"fps_window_ms": 5000}, timeout=30)["fps"], 2) for _ in range(a.measure)] or [0.0]
    if a.frametimes:
        ft = call("frame_times", {"seconds": a.frametimes}, timeout=a.frametimes + 30)
        frame_stats = {k: round(ft[k], 2) for k in ("fps", "p50_ms", "p90_ms", "p99_ms", "max_ms")}
        frame_stats["stutters"] = ft["stutters"]
    status = call("status", {"fps_window_ms": 0}, timeout=10)
    call("screenshot", out=os.path.join(d, "shot.png"), timeout=30)
    summary = {"scene_diff": None if summary_scene is None else round(summary_scene, 1), "name": a.name, "time": datetime.datetime.now().isoformat(timespec="seconds"),
               "rev": git_rev(), "build": status.get("build"), "env": env, "fps": samples,
               "fps_mean": round(statistics.mean(samples), 2), "fps_min": min(samples),
               "fps_max": max(samples)}
    if a.frametimes:
        summary["frames"] = frame_stats
    if walk_stats:
        summary["walk"] = walk_stats
    if a.profile:
        lines = call("log_tail", {"contains": "per second", "limit": 4}, timeout=10)["lines"]
        summary["profile"] = lines
        with open(os.path.join(d, "profile.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(lines))
    for i, pattern in enumerate(a.grep):
        lines = call("log_tail", {"contains": pattern, "limit": 400}, timeout=10)["lines"]
        with open(os.path.join(d, f"log-{i}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# lines containing {pattern!r}\n" + "\n".join(lines) + "\n")
    if a.sample:
        rep = call("sample", {"threads": a.threads, "seconds": a.sample, "max_chars": 400000},
                   timeout=a.sample + 120)
        with open(os.path.join(d, "samples.txt"), "w", encoding="utf-8") as f:
            f.write(rep["report"])
        summary["samples_file"] = rep["file"]
        raw = rep["file"] + ".raw"
        if os.path.exists(raw):
            r = subprocess.run([sys.executable, "-I", os.path.join(REPO, "tools", "nx2", "symbolize.py"),
                                os.path.join(EMU_DIR, a.exe), raw], capture_output=True, text=True)
            with open(os.path.join(d, "samples-sym.txt"), "w", encoding="utf-8") as f:
                f.write(r.stdout + r.stderr)
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    new = not os.path.exists(os.path.join(OUT, "results.tsv"))
    with open(os.path.join(OUT, "results.tsv"), "a", encoding="utf-8") as f:
        if new:
            f.write("time\tname\trev\tmean\tmin\tmax\tenv\n")
        extra = " ".join(f"{k}={v}" for k, v in env.items() if k != "EDEN_NX2_DEBUG_SERVER")
        extra += f" affinity={a.affinity}" if a.affinity else ""
        if a.frametimes:
            extra += " p99={p99_ms}ms max={max_ms}ms stutters={stutters}".format(**frame_stats)
        f.write(f"{summary['time']}\t{a.name}\t{summary['rev']}\t{summary['fps_mean']}\t"
                f"{summary['fps_min']}\t{summary['fps_max']}\t{extra}\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "profile"}, indent=1))
    print("output:", d)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--env", action="append", default=[], help="K=V, repeatable")
    r.add_argument("--profile", action="store_true", help="EDEN_NX2_GPU_PROFILE=1 and keep its report")
    r.add_argument("--sample", type=float, default=0, help="seconds of stack sampling after measuring")
    r.add_argument("--threads", default="GPU,GPUBack,CPUCore_*")
    r.add_argument("--grep", action="append", default=[],
                   help="save the log lines containing this text (e.g. 'DKB gpu_frame'), repeatable")
    r.add_argument("--measure", type=int, default=6, help="number of 5 s fps windows")
    r.add_argument("--settle", type=float, default=10)
    r.add_argument("--walk", type=int, default=0,
                   help="play the scripted route this many times right after loading and report its "
                        "hitches (frames over 50 ms) from the present intervals")
    r.add_argument("--frametimes", type=float, default=0,
                   help="seconds of present intervals to summarise (p50/p90/p99/max, stutters)")
    r.add_argument("--exe", default=os.environ.get("LICHII_EXE", "lichii.exe"))
    r.add_argument("--affinity", help="hex CPU mask for the process (e.g. ffff = the first CCD of a 16-core Ryzen, SMT included)")
    r.add_argument("--slot", type=int, default=int(os.environ.get("OOT_SLOT", "1")),
                   help="save on the Load list, 0 = newest, -1 = the bottom one (default 1)")
    r.add_argument("--attach", action="store_true", help="measure the game already running in gameplay")
    r.add_argument("--kill", action="store_true", help="close a running lichii.exe first")
    sub.add_parser("results")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "results":
        with open(os.path.join(OUT, "results.tsv"), encoding="utf-8") as f:
            print(f.read())
    else:
        run(a)


if __name__ == "__main__":
    main()
