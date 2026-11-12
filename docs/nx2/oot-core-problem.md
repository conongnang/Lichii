# Ocarina of Time: the problem to focus on

Written 2026-11-12 from the code on `oot-amd-perf` (256c2b8972 plus the staged texture cache
work), not from PR claims. One runtime sample is used as a check and is marked as such.

## The problem

**The texture cache gives every guest image a permanent identity: (guest address, exact shape).
Ocarina of Time does not use its render target memory that way.** It rebinds the same memory with
new sizes and formats as it plays, so the cache keeps deciding it has never seen the target
before. Each time it does, it builds a new host image on the GPU back thread, in the middle of
the frame, and keeps the old one.

The code already says this about the game, in three comments in
`src/video_core/texture_cache/texture_cache.h`:

- line 3480: Ocarina's market "makes a new render target every few frames"
- line 3517: Ocarina "creates ~4 such targets a frame outdoors"
- line 3581: Ocarina "resizes a shadow map every frame in parts of Kokiri Forest, at the same
  address: each size is a new depth target and a new texture, both copied from older sizes"

Everything else in this file follows from that mismatch.

## What the code does on each such rebind

All of this runs on the one GPU back thread, holding `buffer_cache.mutex` and
`texture_cache.mutex` (`vk_rasterizer.cpp:1054`), inside a draw or clear.

1. `UpdateRenderTargets` (`texture_cache.h:1414`) calls `FindColorBuffer` / `FindDepthBuffer`,
   then `FindRenderTargetView` (3887), then `FindOrInsertImage` (2844).
2. `FindImage` (2853) only accepts an existing image the new one is a subresource of
   (`IsSubresource`, `util.cpp:1694`). A different size at the same address is a miss.
3. `InsertImage` (3233) calls `JoinImages` (3334). It walks every image overlapping the range,
   sorts them into joins, aliases and "bad overlaps", and then:
   - creates a new host image (3447). That is `vmaCreateImage`, a driver and kernel call
     (`vulkan_memory_allocator.cpp:226`). There is no reuse of a freed image of the same shape.
   - uploads from guest memory, or copies from the older overlapping images (3616-3670).
   - leaves bad overlaps alive, linked to the new image (3550-3562). Their memory is not freed.
4. The new image needs new image views and a new framebuffer.
5. When an old image is finally deleted, `DeleteImage` (4346):
   - ends the fast-draw chain for the whole rasterizer (`Nx2FastDraw::Break`, 4349)
   - marks every render target dirty (4385-4389)
   - removes every framebuffer that used it (`RemoveFramebuffers`, 4483)

## Why this matches the symptoms

- **Stutter in heavy and big places.** More passes and more target shapes mean more images made
  mid-frame. A driver allocation has no time bound, and it blocks the only thread that issues
  draws. Spare CPU cores cannot help, which is why usage stays low while frames are late.
- **Memory fills inside one region, then it stutters harder.** Old shapes pile up as bad overlaps
  until the collector runs (`RunGarbageCollector`, 185). The collector then deletes mid-play,
  which triggers step 5 for each image, and can wait on the GPU for downloads.
- **Fast paths pay less than their authors measured.** Delta draws, the texture memo and the
  framebuffer cache all assume the set of images is stable. Each deletion resets them
  (step 5), so the draws after it take the full `ConfigureImpl` path again
  (`vk_graphics_pipeline.cpp:593`).

Check against a real run (back-thread stack sample, 12 Nov, standing still in Castle Town, the
cheapest case): `TextureCache::InsertImage` 6.3% of samples, of which `vmaCreateImage` 4.4%.
A scene that is not changing should show zero.

## Why the work so far did not pay off

Each of these exists in the tree and each treats one consequence, not the identity model:

| Change | What it removes | What stays |
|---|---|---|
| `ClearOverwritesWhole` (3265) | the upload into a new target a clear overwrites | the new image is still created |
| `copied_whole` in `JoinImages` (3586-3615) | the upload under whole-image copies | the image and the copies |
| Padded-square depth lookup (`FindDepthBuffer`, 3863) | one duplicate shadow map shape | every other shape change |
| `EvictDeadRenderTargets` (536) | some dead targets, earlier | creation; each eviction is a `DeleteImage` |
| Region trim (uncommitted) | old regions' textures after a load | growth inside a region |
| Superseded images (staged, untested) | covered bad overlaps | creation; each delete is a `DeleteImage` |

They make the churn cheaper or clean up after it. None stops a rebind from producing a new
host image, so the gain of each is small and the next scene finds another case.

## What to focus on

Make a render target rebind at known memory cost nothing in steady state.

Done means, for the same scene held still:

1. `Texture cache last 600 frames: created N images, deleted M` reports 0 and 0.
2. `vmaCreateImage` and `InsertImage` are absent from a GPUBack sample.
3. Device memory is flat while staying in one region.
4. `Fast draw ... state changes per second by last break` shows no `invalidation` breaks.

Two parts, in this order:

1. **Stop creating.** A target rebound at an address with a shape seen there before must get
   its earlier host image back, and a new shape must take a recycled host image with the same
   create info instead of calling the driver. The lookup that fails is step 2 above.
2. **Stop the cascade.** Replacing or deleting one image must not reset unrelated state. Today
   one deletion breaks every delta draw and dirties every render target (step 5).

## Not the problem

Do not spend more rounds on these for Ocarina; the code or earlier A/B runs rule them out as
the limit:

- Front/back wait sites in `gpu_split.cpp` and `dma_pusher.cpp`. The back is the slower stage.
- Guest sleep and timer precision, idle waits, upload batching, hot pages.
- Fence behaviour. `gpu_fence_behavior` and `gpu_accuracy` are at defaults, so fences are not
  delayed for the host GPU (`fence_manager.h:86`).

## Measured (2026-11-12, Castle Town test save, `EDEN_NX2_TC_LOG=1`, `HW.GPU:Debug`)

- Code claims above checked against `oot-amd-perf` 256c2b8972: all hold. The region trim is
  committed now (91f7cf7e35).
- Standing still: 19-28 images created and 20-27 deleted per 600 frames, all deleted by the
  collector (normal mode evicts anything unused for 50 frames). The window right after loading:
  ~200-500 created, mostly collector deletes.
- Camera pan (~790 degrees): 3,860-4,310 created and 3,290-3,680 deleted per 600 frames, of which
  the collector only 17-351. Only ~150 are ASTC/BC textures: the rest are render targets.
- Shapes during a pan: the same post-processing chain made again ~32 times per 600 frames each:
  format 105 at 2560x1440 / 1280x720 / 640x360, format 21 at 2560x1440 / 640x360, format 16 and
  format 0 at 1280x720 (96 and 63 times), 384x384 targets. The 2560x1440 and 640x360 format 105
  targets share an address (0x81a139a000). So part 1 is "same addresses, a fixed set of shapes".
- `Fast draw ... by last break` standing: register 96k/s, method 8.7k/s, delta overflow 7.7k/s,
  fence release ~140/s, **no invalidation breaks**. Part 2 (the cascade) is not what costs time
  in this scene.
- `EDEN_NX2_TC_MIN_AGE=600` (collector keeps images 600 frames) changed neither creation during
  the pan nor fps (68.0 vs 68.6, 3 runs each): the pan's deletions are not the collector's.
- Part 1, first step (574fe6a469; **off by default since b402a2bc65**: with it on, play builds
  crashed in amdvlk64.dll inside `vkCreateFramebuffer` for a clear's render targets after 2-4
  minutes of play, and turning it off ended the crashes): the image pool from Fritchii EXP42, extended to
  dedicated allocations, gives a new target a recycled VkImage of the same create info. During the
  pan it served 4,158 of 5,040 image creations (82%); pan 68.9 -> 72.2 fps, worst frame 26-30 ->
  20-24 ms (3 runs each); standing unchanged (68.0 vs 67.7). With it, the pan adds only ~12 ms/s
  of clears, ~10 ms/s of image views and ~19 ms/s of texture uploads to the back thread.
- Still open in part 1: the rebind still goes through `JoinImages`, new views and framebuffers;
  the collector still deletes render targets unused for 50 frames (now cheap to remake).
- Standing frame time is now the back thread's per-draw work: method dispatch ~265 ms/s
  (8.5M methods), stage buffer binds ~129 ms/s (293k), buffer updates ~60 ms/s, plus ~8,000 buffer
  downloads (~71 MB) a second.

## Follow-up checks (2026-11-12, after the pool)

1. **Fence-driven submits.** `EDEN_NX2_FENCE_FLUSH_US` 0 / 500 / 2000: 6,874 / 3,295 / 3,186
   submits a second, stand fps 64.8 / 66.7 / 65.1 (2 runs each, within noise). Fps does not
   follow the window, and 500 -> 2000 barely lowers the count, so most submits are not
   fence-batched ones. Full draws "after a submit" stay ~110/s. The ~8,000 downloads (~70 MB)
   a second show 0 KiB on pages MME DMA reads. Dead as a lever.
2. **Fast-draw register breaks.** By register: `vertex_id_base` (0x446) 68-80k/s of 91-108k,
   `vertex_streams` (0x700) ~10k, `clear_report_value` (0x54c) ~7k, `render_enable` (0x556) ~3k.
   zortos293/Lichii#974 makes the first, third and fourth light: delta draws 12.5k -> 45k/s, but
   full draws got costlier (2.94 -> 3.50 us) and the per-draw average stayed 2.77 -> 2.73 us;
   stand fps 67.1 -> 66.1 (3 runs each). Dropped. The top breaks are then `vertex_streams` 33k/s
   and `vertex_attrib_format` (0x460) 12.5k/s.
3. **Walking.** The perf walk right after loading: 54.1 / 52.8 fps, p99 ~38 ms; the same walk
   after 15 s standing: 57.6 / 57.7, p99 ~26.5 ms. ~4 fps of it is post-load warm-up (standing
   still, fps climbs from ~58 to ~64 over the first ~8 s after a load). The rest is not isolated
   yet: a sideways "circle" test ran into a pillar and was discarded.

## Not verified

- Which shapes churn in Castle Town and how many per frame. The debug lines `created Nx:
  format ... WxH (last at addr)` (`texture_cache.h:787-795`, needs `HW.GPU:Debug`) list them.
  Read these first; they decide whether part 1 is "same address, few sizes" or a real pool.
- How many draws take the delta path, and how many breaks are `invalidation`. The `Fast draw`
  lines print it with `EDEN_NX2_GPU_PROFILE=1`.
- An image pool was ported once (`perf/fritchii-tc`) and showed no fps gain standing still. It
  was not judged on stutter or memory growth, and it kept step 5.
- This targets stutter and memory growth. It is not expected to raise the standing frame rate
  much: that is bounded by per-draw work on the back thread and by guest code speed.
