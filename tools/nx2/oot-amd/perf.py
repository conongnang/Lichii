# OoT performance loop: swap in the daytime Castle Town test save and a fresh test shader cache,
# launch, load, walk, stand and measure, pan the camera ~790 degrees, then close lichii.exe (so it
# doesn't slow the next build). ~80 s a run (~58 s with --warm-cache). Optionally profile.
#
#   perf.py run <name> [--env K=V ...] [--profile] [--sample S] [--threads GPU,GPUBack,CPUCore_*]
#               [--grep TEXT ...] [--measure N] [--walk N] [--no-pan] [--exe lichii.exe]
#               [--attach] [--kill] [--keep] [--no-test-save] [--warm-cache]
#   perf.py results            : the results table
#
# Every switch goes into the launched process's own environment (EDEN_NX2_GPU_PROFILE is read at
# static init, before nx2_switches.txt is parsed). Output: $OOT_PERF_DIR/<name>/ and results.tsv.
# Run with native Windows Python (-I).
import argparse, datetime, filecmp, json, os, shutil, socket, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MCP = os.path.join(REPO, "tools", "nx2", "debug_mcp.py")
OUT = os.environ.get("OOT_PERF_DIR", os.path.join(os.environ["TEMP"], "oot-perf"))
EMU_DIR = os.environ.get("LICHII_DIR", "")  # folder with lichii.exe
ROM = os.environ.get("OOT_ROM", "")  # the game file
REFERENCE = os.environ.get("OOT_REFERENCE", "")  # optional screenshot of the benchmark spot
SERVER = "127.0.0.1:6545"
SWITCHES = os.path.join(os.environ["APPDATA"], "eden", "nx2_switches.txt")
# The Castle Town test save (10:00 daytime, the market's time stands still, so every run sees the
# daytime crowd). Before each launch the live save is compared with it and swapped for it when it
# differs; the live one is kept in oot-test-backup/save-live-<time>/ first.
def find_save_dir():
    """The OoT save of the emulated user that has one (OOT_SAVE_DIR overrides)."""
    if os.environ.get("OOT_SAVE_DIR"):
        return os.environ["OOT_SAVE_DIR"]
    root = os.path.join(os.environ["APPDATA"], "eden", "nand", "user", "save", "0000000000000000")
    users = sorted(os.listdir(root)) if os.path.isdir(root) else []
    for user in users:
        if os.path.isdir(os.path.join(root, user, "0400486024DEC000")):
            return os.path.join(root, user, "0400486024DEC000")
    return os.path.join(root, users[0] if users else "", "0400486024DEC000")


SAVE_DIR = find_save_dir()
BACKUP = os.path.join(os.environ["APPDATA"], "eden", "oot-test-backup")
TEST_SAVE = os.environ.get("OOT_TEST_SAVE", os.path.join(BACKUP, "save-castle-town-20261109",
                                                          "0400486024DEC000"))


CACHE = os.path.join(os.environ["APPDATA"], "eden", "cache")
TEST_CACHE = os.path.join(BACKUP, "shader-test-1k")


SHADERS = os.path.join(CACHE, "shader", "0400486024dec000")


def cache_stamp(exe):
    st = os.stat(os.path.join(EMU_DIR, exe))
    return f"{st.st_size}-{int(st.st_mtime)}"


def unchanged(a, b):
    return os.path.exists(b) and os.path.getsize(a) == os.path.getsize(b) and \
        int(os.path.getmtime(a)) == int(os.path.getmtime(b))


def use_test_cache(exe, warm):
    """Puts the shader cache back to the same state before every run: the ~1.2k-used test cache
    ("cold": boot waits ~25 s building pipelines, then gameplay runs without build hitches). With
    `warm`, from an exe's second run on, also the pipeline binaries and used list its first run
    saved ("baseline"): boot is ~25 s shorter, but those pipelines load during the first ~10 s of
    gameplay with up to ~200 ms hitches (2026-11-12), so the stand numbers are noisier. Only while
    use-test-cache.sh has parked the user's cache in cache/shader-user-live."""
    if not os.path.isdir(os.path.join(CACHE, "shader-user-live")) or not os.path.isdir(TEST_CACHE):
        print("WARNING: the user's shader cache is live (run use-test-cache.sh); boot time and "
              "hitches depend on what earlier runs added", flush=True)
        return "user"
    baseline = os.path.join(OUT, "cache-baseline", cache_stamp(exe))
    if not warm:
        baseline = os.path.join(OUT, "cache-baseline", "none")
    os.makedirs(SHADERS, exist_ok=True)
    keep = set(os.listdir(TEST_CACHE)) | (set(os.listdir(baseline)) if os.path.isdir(baseline) else set())
    for name in os.listdir(SHADERS):
        if name not in keep:
            os.remove(os.path.join(SHADERS, name))
    for name in os.listdir(TEST_CACHE):  # the 900 MB pipeline files never change: copy if needed
        if not unchanged(os.path.join(TEST_CACHE, name), os.path.join(SHADERS, name)):
            shutil.copy2(os.path.join(TEST_CACHE, name), SHADERS)
    if not os.path.isdir(baseline):
        return "cold"
    for name in os.listdir(baseline):
        shutil.copy2(os.path.join(baseline, name), SHADERS)
    return "baseline"


def save_cache_baseline(exe):
    """After an exe's cold run: keep the files that run added or changed as its baseline."""
    baseline = os.path.join(OUT, "cache-baseline", cache_stamp(exe))
    os.makedirs(baseline, exist_ok=True)
    for name in os.listdir(SHADERS):
        path = os.path.join(SHADERS, name)
        if name.endswith(".lock") or unchanged(path, os.path.join(TEST_CACHE, name)):
            continue
        shutil.copy2(path, baseline)


def same_tree(a, b):
    c = filecmp.dircmp(a, b)
    if c.left_only or c.right_only or c.funny_files:
        return False
    if filecmp.cmpfiles(a, b, c.common_files, shallow=False)[1:] != ([], []):
        return False
    return all(same_tree(os.path.join(a, d), os.path.join(b, d)) for d in c.common_dirs)


def use_test_save():
    if not os.path.isdir(TEST_SAVE):
        sys.exit(f"no test save at {TEST_SAVE} (set OOT_TEST_SAVE or pass --no-test-save)")
    if os.path.isdir(SAVE_DIR) and same_tree(SAVE_DIR, TEST_SAVE):
        return
    if os.path.isdir(SAVE_DIR):
        keep = os.path.join(BACKUP, "save-live-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        shutil.copytree(SAVE_DIR, os.path.join(keep, "0400486024DEC000"))
        print(f"live save kept in {keep}; put it back after the runs", flush=True)
        shutil.rmtree(SAVE_DIR)
    shutil.copytree(TEST_SAVE, SAVE_DIR)
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
    for _ in range(360):
        try:
            if call("status", {"fps_window_ms": 0}, timeout=10)["presents"] >= 10:
                return
        except Exception:
            pass
        time.sleep(0.5)
    sys.exit("the game did not start presenting within 3 minutes")


def close():
    """Kill lichii.exe and wait until it is gone, so it doesn't load the PC during the next build."""
    subprocess.run(["taskkill", "/F", "/IM", "lichii.exe"], capture_output=True)
    for _ in range(40):
        if not running():
            return
        time.sleep(0.5)
    print("WARNING: lichii.exe is still running", flush=True)


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


def sky(px):
    """Mean brightness (0-255) of the sky between the start spot's rooftops: ~185 by day."""
    values = [sum(px(x, y)) / 3 for x in range(560, 860, 30) for y in range(8, 40, 8)]
    return sum(values) / len(values)


def on_save_list(px):
    """The Load screen's brown header bar (textured: loose colours)."""
    return all(near(px(x, 30), (95, 75, 45), 30) for x in (300, 980)) and \
        near(px(640, 30), (108, 88, 63), 30)


def slot_selected(px, slot):
    """The selected save card's pulsing white glow, somewhere around its top-left corner."""
    top = 100 + 145 * slot
    return any(sum(px(x, y)) / 3 > 200 for x in range(276, 296, 2) for y in range(top, top + 20, 2))


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
    """A (polled every ~0.3 s, no fixed waits) skips the intro and picks Continue; on the Load list,
    Down `slot` times (0 = newest save), A, A on Load. No button is pressed once loading starts: a
    stray A in gameplay can walk Link into a door. Returns (scene difference, seconds to the list,
    seconds to gameplay)."""
    started = time.time()
    for _ in range(400):
        px = screen()
        # Slot 0 is selected on the list and Load in its dialog: A all the way (the dialog also
        # passes on_save_list, without the list's glow).
        if slot != 0 and on_save_list(px):
            break
        if in_gameplay(px):
            print("booted into gameplay (the game resumed its last spot); measuring there")
            return None, None, round(time.time() - started, 1)
        press("A", 0.15)
    else:
        sys.exit("never reached the save list")
    to_list = round(time.time() - started, 1)
    time.sleep(0.4)  # the list slides in
    # slot -1: the bottom of the list (Down stops at the last save; the list scrolls, so only the
    # scene check after loading can tell it was the right one).
    for _ in range(slot if slot >= 0 else 24):
        press("Down", 0.3)
    for _ in range(8):  # the list slides in and the glow pulses
        if slot < 0 or slot_selected(screen(), slot):
            break
        time.sleep(0.25)
    else:
        sys.exit(f"save slot {slot} is not selected on the Load list")
    press("A", 0.6)
    press("A", 0.0)
    for _ in range(300):
        time.sleep(0.3)
        px = screen()
        if in_gameplay(px):
            to_game = round(time.time() - started, 1)
            if not REFERENCE:
                return 0.0, to_list, to_game
            diff = scene_difference(px)
            if diff > 35:
                print(f"WARNING: scene differs from OOT_REFERENCE (mean diff {diff:.0f}); "
                      "wrong save slot?", flush=True)
            return diff, to_list, to_game
    sys.exit("the save did not load")


# Phases, mapped on screen from the Castle Town test save's start spot (the entrance, facing the
# fountain). Link walks straight at the fountain and stops in the open plaza short of its rim (at
# ~2.75 s he reaches the rim and slides along it, and the running girl can cut in), so the path
# is the same every run. The pan then turns the camera in place: one turn is ~3.4 s of full right
# stick, so 7.5 s is ~790 degrees over the whole plaza. Holds are in wall-clock seconds, not
# presents: FPS Unlock steps the game in real time (10 to 120+ fps), so a present count walks
# further and turns more on a faster build.
WALK = [({"left_stick": [0, 1]}, 2.0)]
PAN = [({"right_stick": [1, 0]}, 7.5)]


def span(seconds):
    """Present-interval stats over the last `seconds` (frame_times `past`; the debug server
    answers one request at a time). Hitches: intervals over 50 ms."""
    ft = call("frame_times", {"past": seconds}, timeout=60)
    return {"seconds": round(seconds, 1), "frames": ft["frames"], "fps": round(ft["fps"], 1),
            "p50_ms": round(ft["p50_ms"], 2), "p99_ms": round(ft["p99_ms"], 2),
            "max_ms": round(ft["max_ms"], 1), "stutters": ft["stutters"], "hitches": ft["hitches"],
            "hitch_ms": ft["hitch_ms"], "worst": ft["worst_ms"], "truncated": ft["truncated"]}


def hold(state, seconds):
    """Holds `state` (buttons/sticks) for `seconds` of wall-clock time: an `input` call that would
    last 10 minutes, released by closing its connection (the server checks every 2 ms)."""
    host, port = SERVER.rsplit(":", 1)
    params = dict(state, frames=36000, timeout_s=900)
    with socket.create_connection((host, int(port)), timeout=10) as s:
        s.sendall(json.dumps({"id": 1, "method": "input", "params": params}).encode() + b"\n")
        time.sleep(seconds)
    time.sleep(0.05)  # let the server release it before the next request


def play(steps, times=1):
    started = time.time()
    for _ in range(times):
        for state, seconds in steps:
            hold(state, seconds)
    return span(time.time() - started)


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
    # The user's switches file (%APPDATA%\eden\nx2_switches.txt) beats the exe folder's: swap in
    # the test's for the run and put the user's back after, so a test never leaves its own.
    saved_switches = None
    if not a.attach and os.path.exists(SWITCHES):
        with open(SWITCHES, encoding="utf-8") as f:
            saved_switches = f.read()
    try:
        run_measured(a, env, d)
    finally:
        if saved_switches is not None:  # read only at startup: safe to restore while it runs
            with open(SWITCHES, "w", encoding="utf-8") as f:
                f.write(saved_switches)


def run_measured(a, env, d):
    cache = None
    if not a.attach and a.test_save:
        if running() and a.kill:
            close()
        if not running():
            use_test_save()
            cache = use_test_cache(a.exe, a.warm_cache)
    try:
        summary = measure(a, env, d)
        summary["cache"] = cache
    finally:
        if not a.attach and not a.keep:
            close()
    if a.warm_cache and cache == "cold" and not running():
        save_cache_baseline(a.exe)
    write_results(a, env, d, summary)


def measure(a, env, d):
    scene, to_list, to_game, boot_s, brightness = None, None, None, None, None
    if not a.attach:
        started = time.time()
        launch(env, a.exe, a.kill, a.affinity)
        boot_s = round(time.time() - started, 1)
        scene, to_list, to_game = load_save(a.slot)
        brightness = round(sky(screen()), 1)
        if brightness < 120:
            print(f"WARNING: the sky is dark ({brightness}): not the daytime test scene", flush=True)
    time.sleep(a.settle)
    # Walk straight after loading (no idle time), stand where it ends (the same view every run),
    # then pan: its end angle varies a little, so nothing is measured after it.
    walk_stats = play(WALK, a.walk) if a.walk else None
    if a.walk:
        call("screenshot", out=os.path.join(d, "walk.png"), timeout=30)
    started = time.time()
    samples = [round(call("status", {"fps_window_ms": 5000}, timeout=30)["fps"], 2) for _ in range(a.measure)] or [0.0]
    frame_stats = span(time.time() - started) if a.measure else None
    pan_stats = play(PAN) if a.pan else None
    if a.frametimes:
        ft = call("frame_times", {"seconds": a.frametimes}, timeout=a.frametimes + 30)
        frame_stats = {k: round(ft[k], 2) for k in ("fps", "p50_ms", "p90_ms", "p99_ms", "max_ms")}
        frame_stats["stutters"] = ft["stutters"]
    status = call("status", {"fps_window_ms": 0}, timeout=10)
    call("screenshot", out=os.path.join(d, "shot.png"), timeout=30)
    summary = {"scene_diff": None if scene is None else round(scene, 1), "name": a.name,
               "time": datetime.datetime.now().isoformat(timespec="seconds"),
               "rev": git_rev(), "build": status.get("build"), "env": env,
               "boot_s": boot_s, "to_list_s": to_list, "to_game_s": to_game, "sky": brightness,
               "fps": samples,
               "fps_mean": round(statistics.mean(samples), 2), "fps_min": min(samples),
               "fps_max": max(samples)}
    if frame_stats:
        summary["frames"] = frame_stats
    if walk_stats:
        summary["walk"] = walk_stats
    if pan_stats:
        summary["pan"] = pan_stats
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
    return summary


def write_results(a, env, d, summary):
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    new = not os.path.exists(os.path.join(OUT, "results.tsv"))
    with open(os.path.join(OUT, "results.tsv"), "a", encoding="utf-8") as f:
        if new:
            f.write("time\tname\trev\tmean\tmin\tmax\tenv\n")
        extra = " ".join(f"{k}={v}" for k, v in env.items() if k != "EDEN_NX2_DEBUG_SERVER")
        extra += f" cache={summary['cache']}" if summary.get("cache") else ""
        extra += f" affinity={a.affinity}" if a.affinity else ""
        if "frames" in summary:
            extra += " p99={p99_ms}ms max={max_ms}ms stutters={stutters}".format(**summary["frames"])
        for phase in ("walk", "pan"):
            if phase in summary:
                extra += " {0}={fps}fps/max{max_ms}ms/{hitches}h".format(phase, **summary[phase])
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
    r.add_argument("--measure", type=int, default=3,
                   help="number of 5 s fps windows standing after the walk")
    r.add_argument("--settle", type=float, default=0, help="idle seconds between loading and the walk")
    r.add_argument("--walk", type=int, default=1,
                   help="times to play the walk right after loading (0 = none); reports its fps, "
                        "p99, max and hitches (frames over 50 ms)")
    r.add_argument("--no-pan", dest="pan", action="store_false",
                   help="skip the ~790 degree camera pan after the walk")
    r.add_argument("--keep", action="store_true", help="leave lichii.exe running after the run")
    r.add_argument("--no-test-save", dest="test_save", action="store_false",
                   help="don't swap in the Castle Town daytime test save and test shader cache")
    r.add_argument("--warm-cache", action="store_true",
                   help="reuse the pipeline binaries of this exe's first run: ~25 s faster boots, "
                        "noisier first stand window")
    r.add_argument("--frametimes", type=float, default=0,
                   help="seconds of present intervals to summarise (p50/p90/p99/max, stutters)")
    r.add_argument("--exe", default=os.environ.get("LICHII_EXE", "lichii.exe"))
    r.add_argument("--affinity", help="hex CPU mask for the process (e.g. ffff = the first CCD of a 16-core Ryzen, SMT included)")
    r.add_argument("--slot", type=int, default=int(os.environ.get("OOT_SLOT", "0")),
                   help="save on the Load list, 0 = newest (the Castle Town test save), -1 = the bottom one")
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
