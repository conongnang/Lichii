# Native graphics pipeline worker emission

Native graphics SPIR-V emission and shader-module creation run on `VkPipelineBuilder` when
an emission reservation is available. Guest-dependent IR translation remains on GPUBack.
Legacy programs and a full reservation budget retain synchronous emission.

## Why the old path still stalls GPUBack

`CurrentGraphicsPipelineSlowPath` calls `CreateGraphicsPipeline` synchronously on a cache miss.
That call reads the shader environments, translates the programs to IR, emits SPIR-V, and creates
Vulkan shader modules. Only the final driver pipeline compilation is queued on the existing
worker pool. The old "Graphics pipeline built on the GPU thread" timing measures the synchronous
part, not the driver compile.

`use_asynchronous_shaders` controls `BuiltPipeline`'s discard policy, not the Vulkan worker pool.
It defaults off. With it on, legacy draws may be discarded when depth is enabled or both draw
counts exceed six. That heuristic tries to preserve small, one-time texture-generation draws.
It does not prove that a draw's output is temporary.

`CanDiscardPendingDraw` therefore rejects discarding **every native draw**, regardless of size or
depth. Native draws may create persistent lighting, storage data, compute inputs, copied render
targets, or readback sources. There is no reliable native classification for safe-to-skip draws.
This track retains that policy whether asynchronous shaders are enabled or disabled.

## What runs where

GPUBack still performs environment reads and IR translation, including the passes that discover
resource bindings. `Configure` needs the resulting `Shader::Info` before it can bind resources.
GPUBack also creates descriptor layouts and allocators, and configures the buffer and texture
caches under their existing rasterizer locks.

Each deferred emission owns its IR pools, programs, runtime information, and stage bindings. The
worker emits SPIR-V, creates shader modules, and performs the existing driver build. It does not
read engine registers, guest memory, buffer or texture caches, or record scheduler commands.
`RenderPassCache::Get` retains its existing internal lock. `PipelineBuild` releases task captures
before publishing completion with a release store. Draws consume it with an acquire load or wait
on its condition variable. All waiters are notified on success and on a worker exception.

GPUBack records the pending draw and can prepare subsequent commands. VulkanWorker waits at that
draw's existing build-completion command, so neither that draw nor later commands overtake the
build. Deferred-emission jobs cannot be claimed for compilation by VulkanWorker. A long build can
still stall command execution and eventually backpressure GPUBack; this is overlap, not free work.

Deferred IR jobs are capped at the existing pipeline worker count. When that budget is full,
the new pipeline uses the original synchronous emission path instead of retaining an unbounded
IR backlog. Up to four released IR pools are retained for reuse. The driver workaround that
restricts broken parallel-compilation devices to one worker is unchanged.

Queued callbacks own the shared build state instead of a raw pipeline pointer. A pipeline already
built by the scheduler can therefore be evicted before its old queued callback runs. Pipeline
destruction waits for running work or finishes queued work before its Vulkan resources disappear.
The pipeline cache drains its workers and explicitly destroys graphics pipelines while the IR
pool cache and other build dependencies are still alive. Destruction may claim a queued native
build; ordinary native draw waits never do.

## Upstream alternative

[Eglantine PR #503](https://github.com/Djihads80/Eglantine/pull/503), inspected at
`7584d618f8dec5fc1fa8c76fde68489ab5f76c9c`, implements full async translation with detached
environments. Its desktop timeout discards native draws, and its texture snapshots still read
live descriptor memory later without synchronization. Porting that behavior would violate this
track's correctness requirements. Making GPUBack wait for the entire translation instead would
move CPU work but leave its serial stall intact.

## Timing and verification

The existing per-pipeline GPU-thread timing remains, with the same pipeline key for correlation.
New `Nx2Perf async_pipelines` lines report worker emission plus driver-build time for slow builds.
The runtime summary separately reports GPU-thread time, GPU-thread graphics emission time,
worker graphics emission time, and time VulkanWorker spent holding draws for pending pipelines.
Emission timing includes shader-module creation. Worker build totals overlap GPUBack work and
must not be added to GPUBack's elapsed time.

`EDEN_NX2_PERF_VERIFY=1` also emits reference SPIR-V on GPUBack. The worker compares every emitted
word against that reference, logs any mismatch as `Nx2Perf async_pipelines:`, and uses the
reference on a mismatch. Verification deliberately adds synchronous work and is not a performance
measurement mode.

The `[async_draw]` tests cover native no-discard behavior and exercise the production
`PipelineBuild` and `EmissionBudget` classes. They cover delayed emission, ordered draw waits
with out-of-order completions, saturation fallback, standard and unknown exceptions, capture
cleanup before completion, stale queued callbacks after eviction, and queued and running builds
during teardown. These are CPU lifecycle tests, not a Vulkan-driver or gameplay replay.

Shader tests compare synchronous and worker emission through the production verification helper.
They include storage writes, shared descriptor-binding numbering, a merged VertexA/VertexB
program with a conditional branch and phi node, preserved stage metadata, and reference selection
on a word mismatch. They do not cover every shader stage or native program form.

The supplied A2 profile has no `CreateGraphicsPipeline`, `TranslateProgram`, or `EmitSPIRV`
samples in its 9,677 GPUBack samples, so it cannot split the reported build cost into translation
and emission. No Ocarina runtime measurements or Windows build are claimed for this cloud change.

The reported 0.2–2.3 seconds per 10 seconds does not establish an FPS improvement. This change
can overlap only the emission fraction with useful GPUBack work, and queue delays, allocation,
and remaining draw holds can reduce that benefit. Its actual savings require
equivalent cold-cache sessions against the original synchronous implementation, not a
warmed-cache second run.

## Skipping draws into redrawn targets (native_async_draws Transient)

With asynchronous shaders on, a native draw whose pipeline is still building waits for it unless
`native_async_draws` (port of Lichii #465: Wait, Bounded, Skip) says otherwise. The default is now
`Transient`: `RasterizerVulkan::PrepareDraw` skips the draw only when the pipeline writes no storage,
transform feedback is off, and every bound color and depth image was fully cleared in this frame and
in the one before (`TextureCache::NoteRenderTargetsCleared`, `RenderTargetsRedrawnReason`), so what the
draw would add is drawn again next frame. Draws into images that persist (HUD and text images, atlases,
impostor mip chains, history buffers) still wait. `EDEN_NX2_NATIVE_ASYNC=wait|bounded|skip|transient`
overrides the setting for comparisons; `EDEN_NX2_ASYNC_DISCARD=1` still forces Skip.

Ocarina of Time, Hyrule Field save, cold Lichii shader cache, `perf.py --walk 2` (RX 9070 XT):

| Mode | Hitches over 50 ms | Hitch time | Worst |
|---|---|---|---|
| Wait | 58-62 | 6.4-7.2 s | 0.25-0.98 s |
| Bounded, 8 ms | 51 | 6.1 s | 0.42 s |
| Transient | 34-38 | 3.2-3.7 s | 0.20-0.31 s |
| Skip | 10-17 | 0.8-1.4 s | 0.15-0.34 s |

Skip leaves the HUD's item labels and D-pad icon missing for good (their images are drawn once, while
the pipeline built); Transient keeps them. What is left is mostly IR translation on the GPU thread
(1.5-1.7 s per 10 s while ~700 new pipelines appear; 500 distinct stage shaders in 847 uses, so
reusing translations would save at most ~40%) and draws into persistent images, which must wait.

## Default is now Skip (2026-11-08, after merging zortos/nx2)

Frame freezes bother more than missing frames, so `native_async_draws` defaults to `Skip`. Same
Hyrule Field save, cold Lichii shader cache, `perf.py --walk 2`, RX 9070 XT, merged build:

| Mode | Hitches over 50 ms | Hitch time | Worst |
|---|---|---|---|
| Transient (draws into persistent targets wait on the driver build) | 23 | 11.4-12.2 s | 1.19-1.22 s |
| Skip | 8 | 0.74 s | 0.18 s |

The pre-merge binary measured the same as the merged one, so the freezes are the held draws, not the merge.
The HUD (hearts, items, map) is intact in the Skip screenshot after the walk; the earlier report of
labels missing for good is not reproduced here. `native_async_draws=Transient` or
`EDEN_NX2_NATIVE_ASYNC=transient` restores the old behaviour.
