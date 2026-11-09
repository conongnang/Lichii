// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR frame generation is not part of this tree. Ocarina's FSR 4 path (ported from
// CheezerWheezer/Lichii) feeds it from its motion data; these no-ops keep that path as it was
// written, with frame generation never requested.

#pragma once

#include <array>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Scheduler;

namespace AmdFrameGen {

[[nodiscard]] inline bool Requested() {
    return false;
}

[[nodiscard]] inline bool SwapchainActive() {
    return false;
}

struct PrepareImage {
    VkImage image;
    VkFormat format;
    VkImageAspectFlags aspect;
};

struct PrepareInputs {
    PrepareImage depth;
    PrepareImage motion_vectors;
    VkExtent2D render_size;
    std::array<f32, 16> view_projection;
    bool reset;
};

inline void Prepare(Scheduler&, const PrepareInputs&) {}

} // namespace AmdFrameGen
} // namespace Vulkan
