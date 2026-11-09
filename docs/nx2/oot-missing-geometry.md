# Ocarina of Time: missing and misplaced static geometry (AMD, 2026-11-09)

Two symptoms, probably one bug: the game stops submitting (or mis-submits) its static world
geometry.

1. **Wrong models after a long session** (Hyrule Field, Kakariko). Huge dark wedges across the sky,
   and in Kakariko a whole house interior (chairs, chandelier, fog) drawn around Link while he
   stands outdoors. Appears after a long play time and flickers; a relaunch clears it.
2. **Sacred Forest Meadow is empty** (save slot 0, boots straight into it). Only Link, Navi, the
   warp pad, distant trees and a floating flower patch are drawn; ground, hedges and walls are
   missing in every direction. Reproduces about 40 s after boot, every time.

## What the draws are

- Symptom 1: the bad draws use vertex shader `79d3cc5c0254b1bd` (static meshes and heightmap
  terrain: it samples R32_FLOAT 260x260 height tiles and BC6H tiles in the vertex stage) with
  fragment shader `292b3848bd8c1dbf`. `reject_shaders` on `79d3cc5c0254b1bd` removes the wedges and
  the fake interior and shows the correct scene behind them.
- Symptom 2: the missing geometry is absent from the depth buffer, so it is never drawn. Only
  ~84 G-buffer draws run (300+ in a normal outdoor frame) and `79d3cc5c0254b1bd` has 3 small draws.
  With `EDEN_DKB_FRAME_TRACE=1` the rasterizer reports ~13-15 draws per frame skipped before
  recording because they have **zero vertices**: non-indexed, `count=0`, `instances=1`, into the
  G-buffer target (`rt0=0x8114001000`, ~11 shader pairs such as vs `0x115db6a600`) and a second
  target (`0x8117667000`). The skip is at `RasterizerVulkan::Draw` (zero-count early return); a
  debug log of these draws was added there under `EDEN_DKB_FRAME_TRACE`. These are the prime
  suspects for the missing meadow: where does their vertex count come from (a macro reading
  GPU-written draw arguments?), and why is it 0?

## Ruled out (no visible change)

| Test | How |
|---|---|
| Stale cached vertex/index data | `capture_frame` with `geometry`: all 168 ranges match guest memory; none zero or unmapped |
| Bad per-object transforms / camera | Decoded c3/c4 of every `79d3` draw: sane matrices, no NaN |
| Bad heightmaps | All R32 height tiles sampled by the vertex stage are in -0.03..0.32 |
| Fast draw, geometry sync skip, UBO reuse, macro memo | Live `switches` off, one at a time |
| Conditional rendering | `render_conditions` switch off (draws everything) |
| Occlusion query results | `occlusion_results` switch off (every report "visible") |
| CPU readback of GPU writes | `EDEN_NX2_BUFFER_DOWNLOADS=1`, then also `EDEN_NX2_READ_TRAPS=0 EDEN_NX2_PREFLUSH=0` |
| Split GPU thread | `EDEN_NX2_SPLIT=0` (zero-count draws remain, ~15/frame) |
| GPU accuracy | High instead of Low (Fast) |
| Built-in patches | FPS Unlock, Dynamic FPS by Dark and Max Resolution all disabled |
| Recent regression | Nov 8 `port/fritchii` build is identical |
| Fork-only bug | Upstream Eglantine `nx2` 0.1.0-rc3 (`78743f30c2`, built locally, portable) is identical on this AMD machine |
| Missing DLC / file reads | The NUP holds only the v1.1.1 update; no read errors |
| Emulator-side drops | `EDEN_NX2_RENDER_TRACE=1`: every traced draw is configured; no skipped, rejected or indirect draws |

Not related: the 8 MiB `Unmapped Device ReadBlock` at ~35 s is a scratch storage buffer of two
late compute dispatches, and upstream issue #798 is the Water Temple, not this scene.

## Next steps

1. Find the source of the zero vertex counts: trace the macro (`EDEN_NX2_MME_LOG` / `MME_RECORD`)
   that issues these draws and what memory it reads, then compare with the compute dispatches that
   write it (`EDEN_NX2_MME_DMA_STATS`).
2. Check whether the same zero-count draws exist in a working scene (Kakariko at boot) and whether
   symptom 1 (wrong models after long play) shows wrong non-zero counts or first vertices from the
   same path.
3. Get an NVIDIA or hardware reference of the meadow to confirm it is AMD-specific.

## Tooling added on `debug/live-geometry-switches`

`switches` (fast_draw, geometry_sync_skip, ubo_reuse, mme_memo, render_conditions,
occlusion_results), `capture_frame` `geometry` (guest vs cached vertex/index bytes) and the
zero-count draw log. See `docs/nx2/debug-mcp.md`.
