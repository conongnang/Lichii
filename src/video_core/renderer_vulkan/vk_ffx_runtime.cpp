// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef _WIN32

#include <mutex>
#include <string>

#include <windows.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_ffx_runtime.h"

namespace Vulkan::FfxRuntime {

namespace {

constexpr const wchar_t* LOADER = L"amd_fidelityfx_loader_vk.dll";
// The loader loads these beside itself; they go with it.
constexpr const wchar_t* PROVIDERS[] = {L"amd_fidelityfx_framegeneration_vk.dll",
                                        L"amd_fidelityfx_upscaler_vk.dll"};

struct Loaded : Api {
    HMODULE module{};
};

std::mutex mutex;
std::weak_ptr<const Loaded> current;

void Unload(const Loaded* loaded) {
    std::scoped_lock lock{mutex};
    FreeLibrary(loaded->module);
    for (const wchar_t* name : PROVIDERS) {
        for (int i = 0; i < 8; ++i) {
            const HMODULE module = GetModuleHandleW(name);
            if (!module || !FreeLibrary(module)) {
                break;
            }
        }
    }
    LOG_INFO(Render_Vulkan, "AMD FidelityFX: runtime unloaded{}",
             FrameGenerationLoaded() ? " (frame generation DLL still loaded)" : "");
    delete loaded;
}

} // Anonymous namespace

std::shared_ptr<const Api> Acquire() {
    std::scoped_lock lock{mutex};
    if (auto existing = current.lock()) {
        return existing;
    }
    wchar_t path[MAX_PATH]{};
    const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring dll(path, length);
    dll = dll.substr(0, dll.find_last_of(L"\\/") + 1) + LOADER;
    const HMODULE module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        LOG_INFO(Render_Vulkan, "AMD FidelityFX: amd_fidelityfx_loader_vk.dll not found next to "
                                "the emulator");
        return nullptr;
    }
    auto* const loaded = new Loaded{};
    loaded->module = module;
    loaded->create_context =
        reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
    loaded->destroy_context =
        reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
    loaded->configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(module, "ffxConfigure"));
    loaded->query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
    loaded->dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
    if (!loaded->create_context || !loaded->destroy_context || !loaded->configure ||
        !loaded->query || !loaded->dispatch) {
        LOG_WARNING(Render_Vulkan, "AMD FidelityFX: the loader lacks its exports");
        FreeLibrary(module);
        delete loaded;
        return nullptr;
    }
    // The deleter takes the lock itself: release happens outside it.
    std::shared_ptr<const Loaded> shared{loaded, [](const Loaded* l) { Unload(l); }};
    current = shared;
    return shared;
}

bool FrameGenerationLoaded() {
    return GetModuleHandleW(PROVIDERS[0]) != nullptr;
}

} // namespace Vulkan::FfxRuntime

#endif // _WIN32
