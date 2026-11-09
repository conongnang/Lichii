// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace Service::Nx2 {

/// A built-in patch's frame hold (FileSys::BuiltinFrameHold) at its guest address: while the
/// byte there is not `free_value`, the nx2:vi2 vsync event runs at 60 Hz whatever the frame rate
/// limit (Ocarina of Time's songs and cutscenes count frames). Address 0: no hold.
struct FrameHold {
    std::atomic<u64> address{0};
    std::atomic<u8> free_value{0};
};

inline FrameHold& GetFrameHold() {
    static FrameHold hold;
    return hold;
}

} // namespace Service::Nx2
