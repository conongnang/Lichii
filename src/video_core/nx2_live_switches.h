// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <string_view>

#include "video_core/nx2_fast_draw.h"

/// GPU shortcuts the debug server's `switches` tool turns off and on while the game runs, so a bug
/// that takes a long session to appear can be A/B tested without a relaunch. Each one is read with a
/// relaxed load where it is used and is safe to flip between any two uses.
namespace VideoCommon::Nx2Live {

namespace Detail {
[[nodiscard]] inline bool EnvOn(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}
} // namespace Detail

/// BufferCache::SynchronizeGeometryBuffer skips scanning a vertex or index range no CPU write
/// reached since its last scan.
inline std::atomic<bool> g_geometry_sync_skip{true};
/// Small uniform buffers with the bytes of the slot's previous bind reuse its stream allocation
/// (uniform_stream_reuse.h). EDEN_NX2_UBOREUSE=0 starts with it off.
inline std::atomic<bool> g_ubo_reuse{Detail::EnvOn("EDEN_NX2_UBOREUSE")};
/// Repeated native macro calls replay their recorded effects (macro_memo.h). EDEN_NX2_MME_MEMO=0
/// starts with it off.
inline std::atomic<bool> g_mme_memo{Detail::EnvOn("EDEN_NX2_MME_MEMO")};

[[nodiscard]] inline bool GeometrySyncSkip() noexcept {
    return g_geometry_sync_skip.load(std::memory_order_relaxed);
}
[[nodiscard]] inline bool UboReuse() noexcept {
    return g_ubo_reuse.load(std::memory_order_relaxed);
}
[[nodiscard]] inline bool MmeMemo() noexcept {
    return g_mme_memo.load(std::memory_order_relaxed);
}

struct Switch {
    std::string_view name;
    std::atomic<bool>* value;
    std::string_view description;
};

[[nodiscard]] inline std::array<Switch, 4> Switches() {
    return {{
        {"fast_draw", &Nx2FastDraw::g_enabled,
         "delta draws that keep the previous draw's pipeline, views and samplers (nx2_fast_draw.h)"},
        {"geometry_sync_skip", &g_geometry_sync_skip,
         "skip re-scanning vertex and index ranges no tracked CPU write reached"},
        {"ubo_reuse", &g_ubo_reuse, "rebind the stream copy of unchanged small uniform buffers"},
        {"mme_memo", &g_mme_memo, "replay the recorded effects of repeated macro calls"},
    }};
}

} // namespace VideoCommon::Nx2Live
