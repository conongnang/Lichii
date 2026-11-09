// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR 4 (FidelityFX SDK 2.4, Vulkan) in place of a game's DLSS evaluation.
//
// The upscaler runs inside AMD's signed DLLs (externals/FidelityFX-SDK-2.4/bin), loaded at runtime
// on Windows. The DLL picks the model for the GPU: the FP8 model on RDNA4, the INT8 model on RDNA3
// and RDNA2 (RX 6000) cards with a current driver. Where it offers no FSR 4 the upscaler reports
// itself unavailable and the caller keeps the built-in FSR 3.1.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

class Fsr4Upscaler {
public:
    struct Image {
        VkImageView view;
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The view's level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
    };

    struct Inputs {
        Image color;          ///< Jittered colour at the render size.
        Image depth;          ///< Device depth (reversed) at the render size.
        Image motion_vectors; ///< Motion to the previous frame at the render size.
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        /// This frame's sample offset in render pixels, as the game hands its own DLSS.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels (the render size for UV offsets).
        std::array<f32, 2> motion_vector_scale;
        f32 frame_time_ms;
        bool reset;
        bool sharpen;
        f32 sharpness; ///< 0 to 1 as FSR takes it.
        bool depth_inverted = true;
        bool hdr = true; ///< Linear HDR colour. False for tonemapped, perceptually encoded colour.
    };

    /// Whether AMD's DLLs are present and offer FSR 4, or else FSR 3.1, for this device (FSR 4
    /// turned on, or AMD frame generation requested). Checked once.
    [[nodiscard]] static bool IsAvailable(const Device& device);
    /// Whether the upscaler IsAvailable picked is FSR 4 (else FSR 3.1 from AMD's runtime).
    [[nodiscard]] static bool IsFsr4(const Device& device);

    /// Throws vk::Exception when FSR 4 can't start on this device.
    explicit Fsr4Upscaler(const Device& device, Scheduler& scheduler,
                          MemoryAllocator& memory_allocator);
    ~Fsr4Upscaler();

    /// Records one evaluation into the output image (RGBA16F at the upscale size, GENERAL
    /// layout). Returns false when FSR 4 can't run, so the caller can fall back to FSR 3.1.
    bool Upscale(const Inputs& inputs);

    [[nodiscard]] bool Failed() const noexcept {
        return failed.load();
    }

    [[nodiscard]] VkImage OutputImage() const noexcept {
        return *output_image;
    }
    [[nodiscard]] VkImageView OutputView() const noexcept {
        return *output_view;
    }
    /// Nearest-neighbour clamp sampler (normalized coordinates), for reading the output.
    [[nodiscard]] VkSampler PointSampler() const noexcept {
        return *point_sampler;
    }

private:
    struct Context;

    void CreateOutput(VkExtent2D upscale_size);
    bool CreateContext(VkExtent2D render_size, VkExtent2D upscale_size, bool depth_inverted,
                       bool hdr);
    void DestroyContext();

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    std::unique_ptr<Context> context;
    VkExtent2D render_extent{};
    VkExtent2D upscale_extent{};
    bool depth_inverted_input = true;
    bool hdr_input = true;
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Sampler point_sampler;
    std::atomic<bool> failed{};
};

} // namespace Vulkan
