// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

#ifdef _WIN32
#include "api/include/ffx_api.h"
#endif

namespace Vulkan::FfxRuntime {

#ifdef _WIN32
/// AMD's FidelityFX API, from amd_fidelityfx_loader_vk.dll next to the emulator.
struct Api {
    PfnFfxCreateContext create_context{};
    PfnFfxDestroyContext destroy_context{};
    PfnFfxConfigure configure{};
    PfnFfxQuery query{};
    PfnFfxDispatch dispatch{};
};

/// Loads AMD's runtime, or shares the one loaded. It unloads (with the upscaler and frame
/// generation DLLs the loader brought in) when the last holder lets go, so the next game loads it
/// afresh: kept loaded, frame generation kept the stopped game's device and the next game crashed.
/// Holders destroy their contexts first. Null when the DLL or its exports are missing.
[[nodiscard]] std::shared_ptr<const Api> Acquire();

/// Whether AMD's frame generation DLL is still in the process (nothing has unloaded it).
[[nodiscard]] bool FrameGenerationLoaded();
#endif

} // namespace Vulkan::FfxRuntime
