// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class DescriptorPool;
class Device;
class MemoryAllocator;
class Scheduler;

/// Where a skinned draw keeps its position, joint weights and joint indices in a vertex.
struct OcarinaVertexLayout {
    u32 stride{};
    VkFormat position{}, weights{}, extra_weights{}, joints{}, extra_joints{};
    u32 position_offset{}, weights_offset{}, extra_weights_offset{}, joints_offset{},
        extra_joints_offset{};
    u32 weight_count{}; ///< Weights used from the first group (joints with them)
    u32 extra_count{};  ///< and from the second (0 without one)
    u32 rigid{};        ///< A solid object: bone 0 is its whole transform, no weights

    bool operator==(const OcarinaVertexLayout&) const = default;
};

struct OcarinaActorMesh {
    OcarinaVertexLayout layout;
    u64 key{};               ///< Its vertex and index buffers; the GPU copy is kept by this key
    std::vector<u8> vertices; ///< Empty when the GPU copy is kept (HasActorMesh)
    std::vector<u16> indices;
    std::array<f32, 4> current_origin;
    std::array<f32, 4> previous_origin;
    std::array<f32, 280 * 12> current_bones;
    std::array<f32, 280 * 12> previous_bones;
};

#ifdef EGLANTINE_STREAMLINE
class OcarinaLiveDlss {
public:
    OcarinaLiveDlss(const Device& device, Scheduler& scheduler, DescriptorPool& pool,
                    ComputePassDescriptorQueue& queue, MemoryAllocator& allocator);
    ~OcarinaLiveDlss();

    [[nodiscard]] bool EvaluationFailed() const;
    /// Save the 1440p DLSS copy before the game adds its HUD.
    void SaveSceneReference(VkImageView color);
    /// Composite the HUD from the guest image over the matching 4K scene.
    void Composite4k(VkImageView native_composite, u64 frame);
    VkImage CompositedImage() const;
    VkImageView CompositedView() const;
    /// Whether a model's vertices and indices are already on the GPU (no need to read them).
    [[nodiscard]] bool HasActorMesh(u64 key) const;
    /// AMD frame generation: writes `output` (rgba8 storage, `extent`) as the presented image
    /// without its HUD, from that image's raw bytes and the scene its final pass read.
    /// AMD frame generation: marks the pixels that differ between FSR 4's output before the
    /// game drew its HUD and the same image at the final pass; the next frame's motion for AMD is
    /// cleared there.
    void MakeHudMask(VkImageView before_hud, VkImageView after_hud, VkExtent2D extent);
    /// The same for the 4K mode (1440p scene, HUD composited over the 4K output): the 1440p scene
    /// saved before the HUD against the game's 1440p composite, which differ slightly everywhere.
    void MakeHudMask4k(VkImageView native_composite);
    /// The next Record builds the HUD mask from these (the last frame's FSR 4 output before its
    /// HUD, and the same image as the final pass left it), before it overwrites the second.
    void SetHudInputs(VkImageView before_hud, VkImageView after_hud);
    void MakeHudless(VkImageView scene, VkImageView presented_raw, VkImage output,
                     VkImageView output_storage, VkExtent2D extent, f32 threshold);

    /// Queue motion, DLSS evaluation and guest writeback for one frame. A false return leaves
    /// the guest dispatch in charge. The scheduler checks EvaluationFailed before skipping it.
    bool Record(VkImageView color, VkImageView depth, VkImageView target, VkExtent2D extent,
                VkExtent2D output_extent, VkExtent2D target_extent, const std::array<f32, 16>& current,
                const std::array<f32, 16>& previous, std::span<const OcarinaActorMesh> meshes,
                VkImage depth_image, u64 frame);

private:
    void MakeHudMaskWith(bool four_k, VkImageView before_hud, VkImageView after_hud,
                         VkExtent2D extent);
    struct Impl;
    std::unique_ptr<Impl> impl;
};
#endif

} // namespace Vulkan
