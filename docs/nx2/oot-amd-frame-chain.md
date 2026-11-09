# Ocarina of Time: what limits the uncapped frame rate (2026-11-07)

Castle Town market (save slot 0), Ryzen 9 9950X3D + RX 9070 XT, AMD 2.0.416, Windows, docked, built-in
Max Resolution add-on, speed limit off, vsync Immediate. Release `oot-amd-perf-2` code.

## Numbers

| Setup | fps | p99 frame |
|---|---|---|
| Default | 60.5-60.9 | 18.5 ms |
| `pin_gpu_threads` on | 66.3-66.5 | 17.2 ms |
| + `cpu_accuracy` Unsafe | 67.0 | 17.1 ms |
| Capped (limiter on) | 60, flat | |

Run-to-run spread is about +-1.5 fps; compare two runs per arm.

## Where a frame goes

`EDEN_NX2_SVC_TIMELINE` + `tools/nx2/svc_timeline.py --sleep-is-run` rebuild each frame's chain of guest
waits and wakes. About 15 ms per frame:

- **~13.5 ms is guest code on the JIT.** The game runs a job system: the presenter (t92), a render thread
  (t88) and ~12 workers (t152-t162), each 20-40% busy, handing work along one chain. Wake latency is
  ~0.2 ms per frame; the guest is never blocked on the present fence (`nx2:vi2 last 600 presents`:
  0 us).
- **~1.4 ms is the presenter polling GPU progress.** It submits the whole frame in one burst at its end
  (8 `SUBMIT_GPFIFO` ioctls in ~0.5 ms), then calls a query that reads an NVN sync object's word
  (`ldr x8,[x0,#8]; ldr w0,[x8]`) and, until it reads 1, sleeps 1 ms and asks again (1-3 times per
  frame). The front and back GPU threads only start a frame once it is submitted, and need about one
  frame time to process it, so they finish just after the next frame's check.

The GPU threads ping-pong (front waits ~50%, back idle ~25%), and the host GPU is ~55% busy. Extra
front-thread work (`EDEN_NX2_CMDPAGE_DIAG`) cost 66 -> 57.5 fps, so the GPU threads' per-frame
latency is on the critical path through the presenter's poll.

## Measured and not kept

- **Front writer waits.** ~10,000 a second block before command list entries and none finds a newly
  GPU-written list (`EDEN_NX2_CMDPAGE_DIAG`: `writer checks ... found written 0`). Trusting pages parsed
  clean in two submissions removed every wait but gained 0 fps (the back is the slower stage) and
  missed 24 pages at load (memory reused for GPU output). Dropped.
- **Native uniform bank compares** (~7% of the back thread). 11% of the changed pages are changes the
  memory tracker missed, so the compare is needed. Skipping it entirely (an unsafe experiment):
  67.6/69.5 fps against 67.6/64.8, within noise.
- **LSE atomics** are emulated as exclusive pairs that end the JIT block, but blocks right after
  atomics are ~1% of guest samples.
- **Guest JIT profile** (`EDEN_NX2_JIT_MAP` + `tools/nx2/jit_profile.py`) is flat: hottest block < 1%;
  dispatch ~1.8%; memory fallbacks ~3%, 1.2% of it spinning on the buffer/texture cache locks in
  `OnCPUWrite`/`GetFlushArea` that the back thread holds.

## Back thread (clean samples)

`ProcessDraw` 46% (`GraphicsPipeline::Configure` 32%, uniform binds 14% of which bank `memcmp` ~7%),
method dispatch 10%, `PrepareDraw` 9%, queries 5%, `FlushCaching` -> `MarkRegionCaching` ->
`VirtualProtect` 3-4%, idle 15-25%.

## If this is picked up again

1. Cut back-thread work per frame (uniform binds, `VirtualProtect` batching in `MarkRegionCaching`,
   query cache lock): each ms is presenter sleep saved, up to ~1.4 ms (~66 -> ~72 fps).
2. Beyond that the guest job chain (~13.5 ms) limits; 120 fps needs guest code ~1.6x faster.
3. Keep `pin_gpu_threads` on: +9% here, because guest cores stay off the GPU threads' cores.
