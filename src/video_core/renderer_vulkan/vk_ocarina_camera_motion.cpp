// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#include <optional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#ifdef EGLANTINE_STREAMLINE
#include "common/logging.h"
#include "video_core/host_shaders/ocarina_dlss_color_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_output_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_4k_composite_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_4k_reference_comp_spv.h"
#include "video_core/host_shaders/ocarina_hudless_comp_spv.h"
#include "video_core/host_shaders/ocarina_hud_mask_4k_comp_spv.h"
#include "video_core/host_shaders/ocarina_hud_mask_comp_spv.h"
#include "video_core/host_shaders/ocarina_hud_motion_comp_spv.h"
#include "video_core/host_shaders/ocarina_hud_depth_comp_spv.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_amd_frame_gen.h"
#include "video_core/renderer_vulkan/vk_fsr4_upscaler.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"

#include "video_core/host_shaders/ocarina_camera_motion_comp_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_frag_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_vert_spv.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

using Matrix = std::array<f32, 16>;

/// Reproject depth from the current clip space directly into the previous camera's clip space.
std::optional<Matrix> ClipToPreviousClip(const Matrix& current, const Matrix& previous) {
    std::array<std::array<f64, 8>, 4> rows{};
    for (size_t i = 0; i < 4; ++i) {
        std::copy_n(current.begin() + i * 4, 4, rows[i].begin());
        rows[i][i + 4] = 1.0;
    }
    for (size_t col = 0; col < 4; ++col) {
        const auto pivot = std::max_element(rows.begin() + col, rows.end(),
            [col](const auto& a, const auto& b) { return std::abs(a[col]) < std::abs(b[col]); });
        std::swap(rows[col], *pivot);
        const f64 divisor = rows[col][col];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-10) {
            return std::nullopt;
        }
        for (f64& value : rows[col]) {
            value /= divisor;
        }
        for (size_t row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const f64 scale = rows[row][col];
            for (size_t i = 0; i < 8; ++i) {
                rows[row][i] -= scale * rows[col][i];
            }
        }
    }
    Matrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            f64 value{};
            for (size_t i = 0; i < 4; ++i) {
                value += previous[row * 4 + i] * rows[i][col + 4];
            }
            result[row * 4 + col] = static_cast<f32>(value);
        }
    }
    if (!std::ranges::all_of(result, [](f32 value) { return std::isfinite(value); })) {
        return std::nullopt;
    }
    return result;
}

constexpr VkFormat MOTION_FORMAT = VK_FORMAT_R16G16_SFLOAT;
constexpr std::array BINDINGS{
    VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
};
constexpr std::array TEMPLATES{
    VkDescriptorUpdateTemplateEntry{0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    0, sizeof(DescriptorUpdateEntry)},
    VkDescriptorUpdateTemplateEntry{1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    sizeof(DescriptorUpdateEntry), sizeof(DescriptorUpdateEntry)},
};
constexpr DescriptorBankInfo BANK{.textures = 1, .images = 1, .score = 2};
constexpr std::array PUSH_CONSTANTS{
    VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Matrix)},
};
constexpr VkImageSubresourceRange MOTION_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

struct ActorTransforms {
    std::array<f32, 16> current;
    std::array<f32, 16> previous;
    std::array<f32, 4> current_origin;
    std::array<f32, 4> previous_origin;
    std::array<f32, 4> extent;
    std::array<u32, 4> counts; ///< Joint weights used: first group, second group
};

vk::Buffer UploadActorData(MemoryAllocator& allocator, std::span<const std::byte> data,
                           VkBufferUsageFlags usage) {
    auto buffer = allocator.CreateBuffer(VkBufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = data.size(),
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    }, MemoryUsage::Upload);
    std::memcpy(buffer.Mapped().data(), data.data(), data.size());
    buffer.Flush();
    return buffer;
}

/// What every model's motion draw shares: the descriptor and pipeline layouts, the render pass into
/// the motion image, the shaders and one pipeline per vertex layout (viewport and scissor dynamic).
class ActorOverlayRenderer {
public:
    explicit ActorOverlayRenderer(const Device& device_) : device{device_} {
        const auto& logical = device.GetLogical();
        const std::array bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT},
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT},
        };
        set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<u32>(bindings.size()), .pBindings = bindings.data(),
        });
        const VkDescriptorSetLayout raw_layout = *set_layout;
        const VkPushConstantRange depth_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 2 * sizeof(u32)};
        layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &raw_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &depth_range,
        });
        const VkAttachmentDescription attachment{
            .format = MOTION_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_GENERAL};
        const VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1, .pColorAttachments = &color,
        };
        render_pass = logical.CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &attachment,
            .subpassCount = 1, .pSubpasses = &subpass,
        });
        vertex_shader = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(OCARINA_ACTOR_MOTION_VERT_SPV), .pCode = std::data(OCARINA_ACTOR_MOTION_VERT_SPV),
        });
        fragment_shader = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(OCARINA_ACTOR_MOTION_FRAG_SPV), .pCode = std::data(OCARINA_ACTOR_MOTION_FRAG_SPV),
        });
    }

    VkDescriptorSetLayout SetLayout() const { return *set_layout; }
    VkPipelineLayout Layout() const { return *layout; }
    VkRenderPass RenderPass() const { return *render_pass; }

    VkPipeline Pipeline(const OcarinaVertexLayout& vertex_layout) {
        for (const auto& [known, pipeline] : pipelines) {
            if (known == vertex_layout) {
                return *pipeline;
            }
        }
        pipelines.emplace_back(vertex_layout, MakePipeline(vertex_layout));
        return *pipelines.back().second;
    }

private:
    vk::Pipeline MakePipeline(const OcarinaVertexLayout& vertex_layout) const {
        const std::array stages{
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, *vertex_shader, "main"},
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, *fragment_shader, "main"},
        };
        const VkVertexInputBindingDescription binding{0, vertex_layout.stride,
                                                      VK_VERTEX_INPUT_RATE_VERTEX};
        // A layout without a second group reads the first again; its weights go unused.
        const std::array attributes{
            VkVertexInputAttributeDescription{0, 0, vertex_layout.position, vertex_layout.position_offset},
            VkVertexInputAttributeDescription{4, 0, vertex_layout.weights, vertex_layout.weights_offset},
            VkVertexInputAttributeDescription{5, 0, vertex_layout.extra_weights,
                                              vertex_layout.extra_weights_offset},
            VkVertexInputAttributeDescription{6, 0, vertex_layout.joints, vertex_layout.joints_offset},
            VkVertexInputAttributeDescription{7, 0, vertex_layout.extra_joints,
                                              vertex_layout.extra_joints_offset},
        };
        const VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data(),
        };
        const VkPipelineInputAssemblyStateCreateInfo assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        const VkPipelineViewportStateCreateInfo viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .scissorCount = 1,
        };
        const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        const VkPipelineDynamicStateCreateInfo dynamic{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
        };
        const VkPipelineRasterizationStateCreateInfo raster{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1,
        };
        const VkPipelineMultisampleStateCreateInfo samples{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        const VkPipelineColorBlendAttachmentState write{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT,
        };
        const VkPipelineColorBlendStateCreateInfo blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &write,
        };
        return device.GetLogical().CreateGraphicsPipeline({
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = static_cast<u32>(stages.size()), .pStages = stages.data(),
            .pVertexInputState = &vertex_input, .pInputAssemblyState = &assembly,
            .pViewportState = &viewport_state, .pRasterizationState = &raster,
            .pMultisampleState = &samples, .pColorBlendState = &blend,
            .pDynamicState = &dynamic,
            .layout = *layout, .renderPass = *render_pass, .basePipelineIndex = -1,
        });
    }

    const Device& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout layout;
    vk::RenderPass render_pass;
    vk::ShaderModule vertex_shader;
    vk::ShaderModule fragment_shader;
    std::vector<std::pair<OcarinaVertexLayout, vk::Pipeline>> pipelines;
};

/// A model's vertices and indices on the GPU, kept while it is drawn.
struct ActorMeshBuffers {
    vk::Buffer vertices;
    vk::Buffer indices;
    u32 index_count{};
    u64 last_frame{};
};

/// One model's motion draw.
struct ActorDraw {
    VkPipeline pipeline;
    VkDescriptorSet set;
    VkBuffer vertices;
    VkBuffer indices;
    u32 index_count;
};

/// Every model's motion draw of a frame, in one render pass.
struct ActorDraws {
    VkRenderPass render_pass{};
    VkFramebuffer framebuffer{};
    VkPipelineLayout layout{};
    std::vector<ActorDraw> draws;
};

class CameraMotionPass final : public ComputePass {
public:
    CameraMotionPass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool)
        : ComputePass(device_, scheduler, pool, BINDINGS, TEMPLATES, BANK, PUSH_CONSTANTS,
                      OCARINA_CAMERA_MOTION_COMP_SPV) {}

    /// Fill camera motion from depth, then overwrite visible actor pixels with skinned motion.
    void Record(Scheduler& scheduler, const void* descriptors, const Matrix& transform,
                VkImage motion, VkExtent2D size, ActorDraws overlays) {
        scheduler.Record([this, descriptors, transform, motion, size,
                          overlays = std::move(overlays)]
                         (vk::CommandBuffer cmdbuf) {
            const VkMemoryBarrier read_depth{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            };
            const VkImageMemoryBarrier init_motion{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = motion,
                .subresourceRange = MOTION_RANGE,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   0, read_depth, {}, init_motion);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, transform);
            cmdbuf.Dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
            const VkMemoryBarrier to_overlay{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, to_overlay);
            if (overlays.draws.empty()) {
                return;
            }
            cmdbuf.BeginRenderPass({
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = overlays.render_pass,
                .framebuffer = overlays.framebuffer,
                .renderArea = {{0, 0}, size},
            }, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport{0, static_cast<f32>(size.height), static_cast<f32>(size.width),
                                      -static_cast<f32>(size.height), 0, 1};
            const VkRect2D scissor{{0, 0}, size};
            cmdbuf.SetViewport(0, viewport);
            cmdbuf.SetScissor(0, scissor);
            // Depth matches in the captured replay leveled off at 16 units in the last place; other
            // GPUs round differently, so a match within 10% of the distance also counts: Link's
            // tunic, hat and hair are shaped by the game beyond his pose, and missed 2%
            // (EDEN_NX2_OOT_LINK_DEPTH=<percent> to compare).
            static const f32 distance_tolerance = [] {
                const char* const value = std::getenv("EDEN_NX2_OOT_LINK_DEPTH");
                return value ? std::strtof(value, nullptr) / 100.0f : 0.10f;
            }();
            struct {
                u32 depth_tolerance;
                f32 distance_tolerance;
            } const comparison{16, distance_tolerance};
            cmdbuf.PushConstants(overlays.layout, VK_SHADER_STAGE_FRAGMENT_BIT, comparison);
            VkPipeline bound{};
            for (const ActorDraw& draw : overlays.draws) {
                if (draw.pipeline != bound) {
                    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipeline);
                    bound = draw.pipeline;
                }
                cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, overlays.layout, 0,
                                          draw.set, {});
                cmdbuf.BindVertexBuffer(0, draw.vertices, 0);
                cmdbuf.BindIndexBuffer(draw.indices, 0, VK_INDEX_TYPE_UINT16);
                cmdbuf.DrawIndexed(draw.index_count, 1, 0, 0, 0);
            }
            cmdbuf.EndRenderPass();
        });
    }
};

constexpr VkImageSubresourceRange COLOR_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
constexpr VkImageSubresourceRange DEPTH_RANGE{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

VkImageMemoryBarrier ProbeBarrier(VkImage image, VkAccessFlags src, VkAccessFlags dst,
                                  VkImageLayout before, VkImageLayout after) {
    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = before,
        .newLayout = after,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = COLOR_RANGE,
    };
}

// The upscaler Ocarina's live replacement runs: NVIDIA DLSS (Streamline) or AMD FSR 4. Both take
// the same inputs; FSR 4 also gets the frame time and the sharpness slider.
class OcarinaUpscaler {
public:
    virtual ~OcarinaUpscaler() = default;
    [[nodiscard]] virtual bool Failed() const = 0;
    [[nodiscard]] virtual VkImage OutputImage() const = 0;
    [[nodiscard]] virtual VkImageView OutputView() const = 0;
    [[nodiscard]] virtual VkSampler PointSampler() const = 0;
    virtual bool Upscale(const StreamlineDlssUpscaler::Inputs& inputs) = 0;
};

class OcarinaDlssUpscaler final : public OcarinaUpscaler {
public:
    OcarinaDlssUpscaler(const Device& device, Scheduler& scheduler, MemoryAllocator& allocator)
        : upscaler{device, scheduler, allocator} {
        LOG_INFO(Render_Vulkan, "Ocarina of Time: the game's upscaler is replaced by NVIDIA DLSS");
    }
    bool Failed() const override {
        return upscaler.Failed();
    }
    VkImage OutputImage() const override {
        return upscaler.OutputImage();
    }
    VkImageView OutputView() const override {
        return upscaler.OutputView();
    }
    VkSampler PointSampler() const override {
        return upscaler.PointSampler();
    }
    bool Upscale(const StreamlineDlssUpscaler::Inputs& in) override {
        return upscaler.Upscale(in);
    }

private:
    StreamlineDlssUpscaler upscaler;
};

class OcarinaFsr4Upscaler final : public OcarinaUpscaler {
public:
    OcarinaFsr4Upscaler(const Device& device, Scheduler& scheduler, MemoryAllocator& allocator)
        : upscaler{device, scheduler, allocator} {
        LOG_INFO(Render_Vulkan, "Ocarina of Time: the game's upscaler is replaced by AMD FSR");
    }
    bool Failed() const override {
        return upscaler.Failed();
    }
    VkImage OutputImage() const override {
        return upscaler.OutputImage();
    }
    VkImageView OutputView() const override {
        return upscaler.OutputView();
    }
    VkSampler PointSampler() const override {
        return upscaler.PointSampler();
    }
    bool Upscale(const StreamlineDlssUpscaler::Inputs& in) override {
        const auto image = [](const StreamlineDlssUpscaler::Image& i) {
            return Fsr4Upscaler::Image{i.view, i.image, i.format, i.range, i.barrier_aspect};
        };
        const auto now = std::chrono::steady_clock::now();
        const f32 frame_time_ms =
            last ? std::chrono::duration<f32, std::milli>(now - *last).count() : 1000.0f / 30.0f;
        last = now;
        // FSR's RCAS, set by the sharpness slider (stops x 100; 200 is off), as for FSR 3.1.
        const int slider = Settings::values.fsr_sharpening_slider.GetValue();
        return upscaler.Upscale({
            .color = image(in.color),
            .depth = image(in.depth),
            .motion_vectors = image(in.motion_vectors),
            .render_size = in.render_size,
            .upscale_size = in.upscale_size,
            .jitter = in.jitter,
            .motion_vector_scale = in.motion_vector_scale,
            .frame_time_ms = frame_time_ms,
            .reset = in.reset,
            .sharpen = slider < 200,
            .sharpness = 1.0f - static_cast<f32>(slider) / 200.0f,
            .depth_inverted = in.depth_inverted,
            .hdr = in.hdr,
        });
    }

private:
    Fsr4Upscaler upscaler;
    std::optional<std::chrono::steady_clock::time_point> last;
};

std::unique_ptr<OcarinaUpscaler> MakeOcarinaUpscaler(const Device& device, Scheduler& scheduler,
                                                     MemoryAllocator& allocator) {
    if (Settings::values.nvidia_dlss.GetValue() && Streamline::IsDlssAvailable()) {
        return std::make_unique<OcarinaDlssUpscaler>(device, scheduler, allocator);
    }
    return std::make_unique<OcarinaFsr4Upscaler>(device, scheduler, allocator);
}

class OcarinaColorPass final : public ComputePass {
public:
    OcarinaColorPass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool,
                      std::span<const u32> code = OCARINA_DLSS_COLOR_COMP_SPV)
        : ComputePass(device_, scheduler, pool, BINDINGS, TEMPLATES, BANK, {},
                      code) {}

    void Record(Scheduler& scheduler, const void* descriptors, VkImage input,
                VkExtent2D extent, const OcarinaUpscaler* require_success = nullptr) {
        scheduler.Record([this, descriptors, input, extent, require_success](vk::CommandBuffer cmd) {
            if (require_success && require_success->Failed()) {
                return;
            }
            const VkImageMemoryBarrier prepare = ProbeBarrier(input, VK_ACCESS_MEMORY_READ_BIT |
                                                                VK_ACCESS_MEMORY_WRITE_BIT,
                                                                VK_ACCESS_SHADER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            const VkMemoryBarrier sample_color{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, sample_color, {}, prepare);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmd.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        });
    }
    void RecordOutput(Scheduler& scheduler, const void* descriptors,
                      const OcarinaUpscaler& upscaler, VkExtent2D extent, u64 frame,
                      bool reset) {
        scheduler.Record([this, descriptors, &upscaler, extent, frame, reset](vk::CommandBuffer cmd) {
            if (upscaler.Failed()) {
                return;
            }
            const VkMemoryBarrier before{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, before);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmd.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            const VkMemoryBarrier after{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, after);
            if (reset) {
                LOG_DEBUG(Render_Vulkan, "Ocarina live DLSS replaced native frame={} history_reset={}", frame, reset);
            }
        });
    }
};

/// A compute pass reading `Samplers` textures and writing one storage image, which keeps its
/// contents when `preserve` is set.
template <u32 Samplers>
class OcarinaImagePass final : public ComputePass {
    static constexpr auto bindings = [] {
        std::array<VkDescriptorSetLayoutBinding, Samplers + 1> result{};
        for (u32 i = 0; i <= Samplers; ++i) {
            result[i] = {i, i == Samplers ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        return result;
    }();
    static constexpr auto templates = [] {
        std::array<VkDescriptorUpdateTemplateEntry, Samplers + 1> result{};
        for (u32 i = 0; i <= Samplers; ++i) {
            result[i] = {i, 0, 1, bindings[i].descriptorType,
                         i * sizeof(DescriptorUpdateEntry), sizeof(DescriptorUpdateEntry)};
        }
        return result;
    }();
    static constexpr DescriptorBankInfo bank{
        .textures = Samplers, .images = 1, .score = Samplers + 1};

public:
    OcarinaImagePass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool,
                     std::span<const u32> code)
        : ComputePass(device_, scheduler, pool, bindings, templates, bank, {}, code) {}

    void Record(Scheduler& scheduler, const void* descriptors, VkImage image, VkExtent2D extent,
                bool preserve) {
        scheduler.Record([this, descriptors, image, extent, preserve](vk::CommandBuffer cmd) {
            const auto output = ProbeBarrier(
                image, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                preserve ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL);
            const VkMemoryBarrier inputs{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, inputs, {}, output);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmd.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            const VkMemoryBarrier written{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, written);
        });
    }
};

class OcarinaHudlessPass final : public ComputePass {
    static constexpr auto bindings = [] {
        std::array<VkDescriptorSetLayoutBinding, 3> result{};
        for (u32 i = 0; i < 3; ++i) {
            result[i] = {i, i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        return result;
    }();
    static constexpr auto templates = [] {
        std::array<VkDescriptorUpdateTemplateEntry, 3> result{};
        for (u32 i = 0; i < 3; ++i) {
            result[i] = {i, 0, 1, bindings[i].descriptorType,
                         i * sizeof(DescriptorUpdateEntry), sizeof(DescriptorUpdateEntry)};
        }
        return result;
    }();
    static constexpr DescriptorBankInfo bank{.textures = 2, .images = 1, .score = 3};
    static constexpr std::array push{VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(f32)}};

public:
    OcarinaHudlessPass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool)
        : ComputePass(device_, scheduler, pool, bindings, templates, bank, push,
                      OCARINA_HUDLESS_COMP_SPV) {}

    void Record(Scheduler& scheduler, const void* descriptors, VkImage image, VkExtent2D extent,
                f32 threshold) {
        scheduler.Record([this, descriptors, image, extent, threshold](vk::CommandBuffer cmd) {
            const auto output = ProbeBarrier(image, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                             VK_IMAGE_LAYOUT_GENERAL);
            const VkMemoryBarrier inputs{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, inputs, {}, output);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmd.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, threshold);
            cmd.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            const VkMemoryBarrier written{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, written);
        });
    }
};

class OcarinaCompositePass final : public ComputePass {
    static constexpr auto bindings = [] {
        std::array<VkDescriptorSetLayoutBinding, 4> result{};
        for (u32 i = 0; i < 4; ++i) {
            result[i] = {i, i == 3 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        return result;
    }();
    static constexpr auto templates = [] {
        std::array<VkDescriptorUpdateTemplateEntry, 4> result{};
        for (u32 i = 0; i < 4; ++i) {
            result[i] = {i, 0, 1, bindings[i].descriptorType,
                         i * sizeof(DescriptorUpdateEntry), sizeof(DescriptorUpdateEntry)};
        }
        return result;
    }();
    static constexpr DescriptorBankInfo bank{.textures = 3, .images = 1, .score = 4};
    static constexpr std::array push{VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(u32)}};

public:
    OcarinaCompositePass(const Device& device_, Scheduler& scheduler, DescriptorPool& pool)
        : ComputePass(device_, scheduler, pool, bindings, templates, bank, push,
                      OCARINA_DLSS_4K_COMPOSITE_COMP_SPV) {}

    void Record(Scheduler& scheduler, const void* descriptors, VkImage image,
                const OcarinaUpscaler& upscaler, u64 frame) {
        scheduler.Record([this, descriptors, image, &upscaler, frame](vk::CommandBuffer cmd) {
            const auto output = ProbeBarrier(image, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                             VK_IMAGE_LAYOUT_GENERAL);
            const VkMemoryBarrier inputs{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, inputs, {}, output);
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptors);
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            const u32 enabled = !upscaler.Failed();
            cmd.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, enabled);
            cmd.Dispatch(3840 / 8, 2160 / 8, 1);
            const VkMemoryBarrier complete{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, complete);
            if (enabled && (!logged || frame % 120 == 0)) {
                LOG_INFO(Render_Vulkan, "Ocarina DLSS presented 3840x2160 frame={}", frame);
                logged = true;
            }
        });
    }

private:
    bool logged{};
};

constexpr VkFormat LIVE_COLOR_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

vk::Image CreateOcarinaImage(MemoryAllocator& allocator, VkExtent2D extent, VkFormat format,
                            VkImageUsageFlags usage) {
    return allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    });
}

vk::ImageView CreateOcarinaView(const Device& device, VkImage image, VkFormat format) {
    return device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = COLOR_RANGE,
    });
}

/// Where Ocarina of Time draws its HUD, as fractions of the screen (x0, y0, x1, y1): hearts and
/// magic, the button icons below them, the item box and the minimap, with a margin.
/// EDEN_NX2_AMD_FG_HUD_RECTS="x0,y0,x1,y1;..." replaces them ("none" for no rectangles).
std::vector<std::array<f32, 4>> HudRects() {
    // None by default: the per-pixel HUD mask (MakeHudMask) keeps the HUD still without
    // stopping the scene around it. For reference, where the HUD sits:
    // {0.015, 0.035, 0.165, 0.145} hearts and magic, {0.005, 0.195, 0.115, 0.290} ocarina,
    // buttons and Navi, {0.870, 0.035, 0.970, 0.235} item box, {0.765, 0.690, 0.995, 0.990} map.
    std::vector<std::array<f32, 4>> rects;
    const char* const value = std::getenv("EDEN_NX2_AMD_FG_HUD_RECTS");
    if (!value) {
        return rects;
    }
    rects.clear();
    std::string_view text{value};
    while (!text.empty()) {
        const size_t end = text.find(';');
        const std::string entry{text.substr(0, end)};
        std::array<f32, 4> rect{};
        if (std::sscanf(entry.c_str(), "%f,%f,%f,%f", &rect[0], &rect[1], &rect[2], &rect[3]) == 4) {
            rects.push_back(rect);
        }
        if (end == std::string_view::npos) {
            break;
        }
        text.remove_prefix(end + 1);
    }
    return rects;
}

/// Clears the HUD rectangles of `motion` (GENERAL layout) by copying from `zero`, an all-zero
/// image of `zero_size`.
void ClearHudMotion(Scheduler& scheduler, VkImage motion, VkExtent2D extent, VkImage zero,
                    VkExtent2D zero_size) {
    static const std::vector<std::array<f32, 4>> rects = HudRects();
    std::vector<VkImageCopy> copies;
    for (const auto& rect : rects) {
        const auto at = [](f32 fraction, u32 size) {
            return static_cast<u32>(std::clamp(fraction, 0.0f, 1.0f) * static_cast<f32>(size));
        };
        const u32 x0 = at(rect[0], extent.width);
        const u32 y0 = at(rect[1], extent.height);
        const u32 x1 = at(rect[2], extent.width);
        const u32 y1 = at(rect[3], extent.height);
        // Tiles of the zero image cover rectangles larger than it.
        for (u32 y = y0; y < y1; y += zero_size.height) {
            for (u32 x = x0; x < x1; x += zero_size.width) {
                copies.push_back(VkImageCopy{
                    .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    .srcOffset = {},
                    .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    .dstOffset = {static_cast<s32>(x), static_cast<s32>(y), 0},
                    .extent = {(std::min)(zero_size.width, x1 - x),
                               (std::min)(zero_size.height, y1 - y), 1},
                });
            }
        }
    }
    if (copies.empty()) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([motion, zero, copies = std::move(copies)](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        cmdbuf.CopyImage(zero, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, motion,
                         VK_IMAGE_LAYOUT_GENERAL, copies);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, after);
    });
}

struct OcarinaFrameSlot {
    OcarinaFrameSlot(const Device& device, MemoryAllocator& allocator, VkExtent2D extent_)
        : extent{extent_}, motion{CreateOcarinaImage(
                              allocator, extent, MOTION_FORMAT, VK_IMAGE_USAGE_STORAGE_BIT |
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT)},
          motion_view{CreateOcarinaView(device, *motion, MOTION_FORMAT)},
          color{CreateOcarinaImage(allocator, extent, LIVE_COLOR_FORMAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)},
          color_view{CreateOcarinaView(device, *color, LIVE_COLOR_FORMAT)} {}

    VkExtent2D extent;
    u64 tick{};
    vk::Image motion;
    vk::ImageView motion_view;
    vk::Image color;
    vk::ImageView color_view;
    /// AMD frame generation's depth: the scene's, with the HUD nearest.
    vk::Image frame_gen_depth;
    vk::ImageView frame_gen_depth_view;
    // Motion of the frame's models: one buffer of poses and transforms, a set per model.
    vk::Buffer actor_data;
    size_t actor_data_size{};
    vk::DescriptorPool actor_pool;
    vk::DescriptorSets actor_sets;
    vk::Framebuffer actor_framebuffer;
};

} // namespace

struct OcarinaLiveDlss::Impl {
    Impl(const Device& device_, Scheduler& scheduler_, DescriptorPool& pool,
         ComputePassDescriptorQueue& queue_, MemoryAllocator& allocator_)
        : device{device_}, scheduler{scheduler_}, queue{queue_}, allocator{allocator_}, pool{pool},
          upscaler{MakeOcarinaUpscaler(device, scheduler, allocator)}, motion_pass{device, scheduler, pool},
          color_pass{device, scheduler, pool},
          output_pass{device, scheduler, pool, OCARINA_DLSS_OUTPUT_COMP_SPV} {}

    const Device& device;
    Scheduler& scheduler;
    ComputePassDescriptorQueue& queue;
    MemoryAllocator& allocator;
    DescriptorPool& pool;
    std::unique_ptr<OcarinaUpscaler> upscaler;
    CameraMotionPass motion_pass;
    OcarinaColorPass color_pass;
    OcarinaColorPass output_pass;
    std::array<std::unique_ptr<OcarinaFrameSlot>, 4> slots;
    std::unique_ptr<ActorOverlayRenderer> actor_renderer;
    std::unordered_map<u64, ActorMeshBuffers> actor_meshes;
    u64 actor_frame{};
    std::optional<u64> last_frame;
    Matrix last_view_projection{};
    vk::Image native_scene;
    vk::ImageView native_scene_view;
    vk::Image composite;
    vk::ImageView composite_view;
    std::unique_ptr<OcarinaColorPass> reference_pass;
    std::unique_ptr<OcarinaCompositePass> composite_pass;
    std::unique_ptr<OcarinaHudlessPass> hudless_pass;
    /// All zero, in the motion format: copied over the HUD's motion for AMD frame generation.
    vk::Image hud_zero;
    vk::Image hud_mask;
    vk::ImageView hud_mask_view;
    VkExtent2D hud_mask_extent{};
    bool hud_mask_ready{};
    std::unique_ptr<OcarinaImagePass<2>> hud_mask_pass;
    std::unique_ptr<OcarinaImagePass<2>> hud_mask_4k_pass;
    std::unique_ptr<OcarinaImagePass<1>> hud_motion_pass;
    std::unique_ptr<OcarinaImagePass<2>> hud_depth_pass;
    VkImageView hud_before{};
    VkImageView hud_after{};
    static constexpr VkExtent2D HUD_ZERO_SIZE{256, 256};

    VkImage HudZero() {
        if (hud_zero) {
            return *hud_zero;
        }
        hud_zero = CreateOcarinaImage(allocator, HUD_ZERO_SIZE, MOTION_FORMAT,
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([image = *hud_zero](vk::CommandBuffer cmdbuf) {
            const auto barrier = [image](VkAccessFlags src, VkAccessFlags dst, VkImageLayout from,
                                         VkImageLayout to) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = src,
                    .dstAccessMask = dst,
                    .oldLayout = from,
                    .newLayout = to,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image,
                    .subresourceRange = MOTION_RANGE,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {},
                                   barrier(0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                           VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
            const VkClearColorValue zero{};
            cmdbuf.ClearColorImage(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, zero, MOTION_RANGE);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   0, {}, {},
                                   barrier(VK_ACCESS_TRANSFER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_READ_BIT,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
        });
        return *hud_zero;
    }
    vk::Sampler linear_sampler;
    std::optional<u64> composited_frame;

    VkSampler LinearSampler() {
        if (!linear_sampler) {
            linear_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .maxAnisotropy = 1.0f,
            });
        }
        return *linear_sampler;
    }
};

OcarinaLiveDlss::OcarinaLiveDlss(const Device& device, Scheduler& scheduler, DescriptorPool& pool,
                                 ComputePassDescriptorQueue& queue, MemoryAllocator& allocator)
    : impl{std::make_unique<Impl>(device, scheduler, pool, queue, allocator)} {}

OcarinaLiveDlss::~OcarinaLiveDlss() = default;

bool OcarinaLiveDlss::EvaluationFailed() const {
    return impl->upscaler->Failed();
}

void OcarinaLiveDlss::SaveSceneReference(VkImageView color) {
    auto& state = *impl;
    constexpr VkExtent2D extent{2560, 1440};
    if (!state.native_scene_view) {
        constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        state.native_scene = CreateOcarinaImage(state.allocator, extent, format,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        state.native_scene_view = CreateOcarinaView(state.device, *state.native_scene, format);
        state.reference_pass = std::make_unique<OcarinaColorPass>(
            state.device, state.scheduler, state.pool, OCARINA_DLSS_4K_REFERENCE_COMP_SPV);
    }
    state.queue.Acquire(state.scheduler, 2);
    state.queue.AddSampledImage(color, state.upscaler->PointSampler());
    state.queue.AddImage(*state.native_scene_view);
    state.reference_pass->Record(state.scheduler, state.queue.UpdateData(), *state.native_scene,
                                 extent, state.upscaler.get());
    state.scheduler.InvalidateState();
}

void OcarinaLiveDlss::SetHudInputs(VkImageView before_hud, VkImageView after_hud) {
    impl->hud_before = before_hud;
    impl->hud_after = after_hud;
}

void OcarinaLiveDlss::MakeHudMask4k(VkImageView native_composite) {
    auto& state = *impl;
    if (!state.native_scene_view || !native_composite) {
        return;
    }
    MakeHudMaskWith(true, *state.native_scene_view, native_composite, {2560, 1440});
}

void OcarinaLiveDlss::MakeHudMask(VkImageView before_hud, VkImageView after_hud,
                                  VkExtent2D extent) {
    MakeHudMaskWith(false, before_hud, after_hud, extent);
}

void OcarinaLiveDlss::MakeHudMaskWith(bool four_k, VkImageView before_hud, VkImageView after_hud,
                                      VkExtent2D extent) {
    auto& state = *impl;
    auto& pass_slot = four_k ? state.hud_mask_4k_pass : state.hud_mask_pass;
    if (!pass_slot) {
        pass_slot = std::make_unique<OcarinaImagePass<2>>(
            state.device, state.scheduler, state.pool,
            four_k ? std::span<const u32>{OCARINA_HUD_MASK_4K_COMP_SPV}
                   : std::span<const u32>{OCARINA_HUD_MASK_COMP_SPV});
    }
    auto& pass = *pass_slot;
    if (!state.hud_mask || state.hud_mask_extent.width != extent.width ||
        state.hud_mask_extent.height != extent.height) {
        state.scheduler.Finish();
        state.hud_mask = CreateOcarinaImage(state.allocator, extent, VK_FORMAT_R8_UNORM,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        state.hud_mask_view = CreateOcarinaView(state.device, *state.hud_mask, VK_FORMAT_R8_UNORM);
        state.hud_mask_extent = extent;
        state.hud_mask_ready = false;
    }
    state.scheduler.RequestOutsideRenderPassOperationContext();
    state.queue.Acquire(state.scheduler, 3);
    state.queue.AddSampledImage(before_hud, state.upscaler->PointSampler());
    state.queue.AddSampledImage(after_hud, state.upscaler->PointSampler());
    state.queue.AddImage(*state.hud_mask_view);
    pass.Record(state.scheduler, state.queue.UpdateData(), *state.hud_mask, extent, false);
    state.scheduler.InvalidateState();
    if (!std::exchange(state.hud_mask_ready, true)) {
        LOG_INFO(Render_Vulkan, "AMD frame generation: the HUD is masked per pixel ({}x{})",
                 extent.width, extent.height);
    }
}

void OcarinaLiveDlss::MakeHudless(VkImageView scene, VkImageView presented_raw, VkImage output,
                                  VkImageView output_storage, VkExtent2D extent, f32 threshold) {
    auto& state = *impl;
    if (!state.hudless_pass) {
        state.hudless_pass =
            std::make_unique<OcarinaHudlessPass>(state.device, state.scheduler, state.pool);
    }
    state.scheduler.RequestOutsideRenderPassOperationContext();
    state.queue.Acquire(state.scheduler, 3);
    state.queue.AddSampledImage(scene, state.LinearSampler());
    state.queue.AddSampledImage(presented_raw, state.upscaler->PointSampler());
    state.queue.AddImage(output_storage);
    state.hudless_pass->Record(state.scheduler, state.queue.UpdateData(), output, extent, threshold);
    state.scheduler.InvalidateState();
}

void OcarinaLiveDlss::Composite4k(VkImageView native_composite, u64 frame) {
    auto& state = *impl;
    if (state.composited_frame == frame) {
        return;
    }
    if (!state.composite_pass) {
        constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        state.composite = CreateOcarinaImage(state.allocator, {3840, 2160}, format,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        state.composite_view = CreateOcarinaView(state.device, *state.composite, format);
        state.composite_pass = std::make_unique<OcarinaCompositePass>(
            state.device, state.scheduler, state.pool);
    }
    state.scheduler.RequestOutsideRenderPassOperationContext();
    state.queue.Acquire(state.scheduler, 4);
    const auto sampler = state.LinearSampler();
    state.queue.AddSampledImage(state.upscaler->OutputView(), sampler);
    state.queue.AddSampledImage(*state.native_scene_view, sampler);
    state.queue.AddSampledImage(native_composite, sampler);
    state.queue.AddImage(*state.composite_view);
    state.composite_pass->Record(state.scheduler, state.queue.UpdateData(), *state.composite,
                                 *state.upscaler, frame);
    state.scheduler.InvalidateState();
    state.composited_frame = frame;
}

VkImage OcarinaLiveDlss::CompositedImage() const {
    return *impl->composite;
}

bool OcarinaLiveDlss::HasActorMesh(u64 key) const {
    return impl->actor_meshes.contains(key);
}

VkImageView OcarinaLiveDlss::CompositedView() const {
    return *impl->composite_view;
}

bool OcarinaLiveDlss::Record(VkImageView color, VkImageView depth, VkImageView target, VkExtent2D extent,
                            VkExtent2D output_extent, VkExtent2D target_extent,
                            const Matrix& current, const Matrix& previous,
                            std::span<const OcarinaActorMesh> meshes, VkImage depth_image, u64 frame) {
    auto& state = *impl;
    if (state.last_frame == frame || state.upscaler->Failed()) {
        return false;
    }
    const auto transform = ClipToPreviousClip(current, previous);
    if (!transform) {
        return false;
    }
    state.scheduler.GetMasterSemaphore().Refresh();
    const auto slot = std::ranges::find_if(state.slots, [&](const auto& entry) {
        return !entry || state.scheduler.IsFree(entry->tick);
    });
    if (slot == state.slots.end()) {
        return false;
    }
    if (!*slot || (*slot)->extent.width != extent.width || (*slot)->extent.height != extent.height) {
        *slot = std::make_unique<OcarinaFrameSlot>(state.device, state.allocator, extent);
    }
    auto& resources = **slot;
    const VkSampler sampler = state.upscaler->PointSampler();
    // Characters and moving objects drawn this frame and the last move by their own poses;
    // everything else, and everything in dialogue, cutscenes and first person, moves with the
    // camera alone.
    ActorDraws overlays;
    ++state.actor_frame;
    if (!meshes.empty()) {
        constexpr u32 MaxDraws = 128; // 64 characters and 64 moving objects
        if (!state.actor_renderer) {
            state.actor_renderer = std::make_unique<ActorOverlayRenderer>(state.device);
        }
        auto& renderer = *state.actor_renderer;
        const auto& logical = state.device.GetLogical();
        if (!resources.actor_pool) {
            const std::array sizes{
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * MaxDraws},
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MaxDraws},
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MaxDraws},
            };
            resources.actor_pool = logical.CreateDescriptorPool({
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = MaxDraws, .poolSizeCount = static_cast<u32>(sizes.size()),
                .pPoolSizes = sizes.data(),
            });
            const std::vector<VkDescriptorSetLayout> layouts(MaxDraws, renderer.SetLayout());
            resources.actor_sets = resources.actor_pool.Allocate({
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = *resources.actor_pool, .descriptorSetCount = MaxDraws,
                .pSetLayouts = layouts.data(),
            });
            const VkImageView motion_view = *resources.motion_view;
            resources.actor_framebuffer = logical.CreateFramebuffer({
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderer.RenderPass(), .attachmentCount = 1,
                .pAttachments = &motion_view,
                .width = extent.width, .height = extent.height, .layers = 1,
            });
        }
        // Poses: a solid object needs one bone, a character all of them.
        const VkDeviceSize alignment = std::max<VkDeviceSize>(
            {state.device.GetStorageBufferAlignment(), state.device.GetUniformBufferAlignment(), 16});
        const auto aligned = [alignment](size_t size) {
            return static_cast<size_t>((size + alignment - 1) / alignment * alignment);
        };
        const auto pose_bytes = [](const OcarinaActorMesh& mesh) {
            return mesh.layout.rigid ? 12 * sizeof(f32) : sizeof(mesh.current_bones);
        };
        size_t needed = 0;
        for (size_t index = 0; index < meshes.size() && index < MaxDraws; ++index) {
            needed += 2 * aligned(pose_bytes(meshes[index])) + aligned(sizeof(ActorTransforms));
        }
        if (resources.actor_data_size < needed) {
            resources.actor_data_size = needed * 2;
            resources.actor_data = state.allocator.CreateBuffer(VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = resources.actor_data_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            }, MemoryUsage::Upload);
        }
        u8* const mapped = reinterpret_cast<u8*>(resources.actor_data.Mapped().data());
        size_t offset = 0;
        const auto put = [&](const void* data, size_t size) {
            const size_t at = offset;
            std::memcpy(mapped + at, data, size);
            offset += aligned(size);
            return VkDescriptorBufferInfo{*resources.actor_data, at, size};
        };
        overlays.render_pass = renderer.RenderPass();
        overlays.framebuffer = *resources.actor_framebuffer;
        overlays.layout = renderer.Layout();
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<std::array<VkDescriptorBufferInfo, 3>> buffer_infos;
        buffer_infos.reserve(MaxDraws);
        const VkDescriptorImageInfo depth_info{sampler, depth, VK_IMAGE_LAYOUT_GENERAL};
        for (const OcarinaActorMesh& mesh : meshes) {
            if (overlays.draws.size() >= MaxDraws) {
                break;
            }
            auto& buffers = state.actor_meshes[mesh.key];
            if (!buffers.vertices) {
                if (mesh.vertices.empty() || mesh.indices.empty()) {
                    state.actor_meshes.erase(mesh.key);
                    continue; // not kept yet, and not read
                }
                buffers.vertices = UploadActorData(state.allocator,
                                                   std::as_bytes(std::span{mesh.vertices}),
                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
                buffers.indices = UploadActorData(state.allocator,
                                                  std::as_bytes(std::span{mesh.indices}),
                                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
                buffers.index_count = static_cast<u32>(mesh.indices.size());
            }
            buffers.last_frame = state.actor_frame;
            const ActorTransforms transforms{
                current, previous, mesh.current_origin, mesh.previous_origin,
                {static_cast<f32>(extent.width), static_cast<f32>(extent.height), 0, 0},
                {mesh.layout.weight_count, mesh.layout.extra_count, mesh.layout.rigid, 0},
            };
            const size_t bones = pose_bytes(mesh);
            buffer_infos.push_back({put(mesh.current_bones.data(), bones),
                                    put(mesh.previous_bones.data(), bones),
                                    put(&transforms, sizeof(transforms))});
            const VkDescriptorSet set = resources.actor_sets[overlays.draws.size()];
            const auto& info = buffer_infos.back();
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info[0]});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info[1]});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1,
                              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &info[2]});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1,
                              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depth_info});
            overlays.draws.push_back({renderer.Pipeline(mesh.layout), set, *buffers.vertices,
                                      *buffers.indices, buffers.index_count});
        }
        resources.actor_data.Flush();
        if (!writes.empty()) {
            logical.UpdateDescriptorSets(writes, {});
        }
    }
    // Models not drawn for a while: frames in flight are long done with them.
    if (state.actor_frame % 120 == 0) {
        std::erase_if(state.actor_meshes, [&](const auto& entry) {
            return entry.second.last_frame + 600 < state.actor_frame;
        });
    }

    state.scheduler.RequestOutsideRenderPassOperationContext();
    state.queue.Acquire(state.scheduler, 2);
    state.queue.AddSampledImage(depth, sampler);
    state.queue.AddImage(*resources.motion_view);
    resources.tick = state.scheduler.CurrentTick();
    state.motion_pass.Record(state.scheduler, state.queue.UpdateData(), *transform,
                             *resources.motion, extent, std::move(overlays));
    state.queue.Acquire(state.scheduler, 2);
    state.queue.AddSampledImage(color, sampler);
    state.queue.AddImage(*resources.color_view);
    state.color_pass.Record(state.scheduler, state.queue.UpdateData(), *resources.color, extent);

    const bool reset = !state.last_frame || frame != *state.last_frame + 1 ||
                       !std::ranges::equal(previous, state.last_view_projection,
                                           [](f32 a, f32 b) { return std::abs(a - b) < 0.001f; });
    if (!state.upscaler->Upscale({
            .color = {*resources.color_view, *resources.color, LIVE_COLOR_FORMAT, COLOR_RANGE,
                      VK_IMAGE_ASPECT_COLOR_BIT},
            .depth = {depth, depth_image, VK_FORMAT_D32_SFLOAT, DEPTH_RANGE, VK_IMAGE_ASPECT_DEPTH_BIT},
            .motion_vectors = {*resources.motion_view, *resources.motion, MOTION_FORMAT,
                               COLOR_RANGE, VK_IMAGE_ASPECT_COLOR_BIT},
            .render_size = extent,
            .upscale_size = output_extent,
            .jitter = {0.0f, 0.0f},
            .motion_vector_scale = {1.0f, 1.0f},
            .reset = reset,
            .depth_inverted = false,
            .hdr = false,
        })) {
        state.scheduler.InvalidateState();
        return false;
    }
    // AMD frame generation reads the same depth and motion for the frame it generates, without
    // motion where the HUD sits (FSR 4 above needed the scene's motion there).
    bool hud_depth = false;
    if (AmdFrameGen::Requested()) {
        if (state.hud_before && state.hud_after) {
            MakeHudMask(state.hud_before, state.hud_after, extent);
        }
        state.hud_before = VK_NULL_HANDLE;
        state.hud_after = VK_NULL_HANDLE;
        // The mask is at the output size, the motion and depth at the scene's (smaller when the
        // game lowers its resolution and FSR upscales): the passes scale between the two.
        if (state.hud_mask_ready) {
            if (!state.hud_motion_pass) {
                state.hud_motion_pass = std::make_unique<OcarinaImagePass<1>>(
                    state.device, state.scheduler, state.pool, OCARINA_HUD_MOTION_COMP_SPV);
            }
            state.scheduler.RequestOutsideRenderPassOperationContext();
            state.queue.Acquire(state.scheduler, 2);
            state.queue.AddSampledImage(*state.hud_mask_view, state.upscaler->PointSampler());
            state.queue.AddImage(*resources.motion_view);
            state.hud_motion_pass->Record(state.scheduler, state.queue.UpdateData(),
                                          *resources.motion, extent, true);
            if (!resources.frame_gen_depth) {
                resources.frame_gen_depth =
                    CreateOcarinaImage(state.allocator, extent, VK_FORMAT_R32_SFLOAT,
                                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
                resources.frame_gen_depth_view = CreateOcarinaView(
                    state.device, *resources.frame_gen_depth, VK_FORMAT_R32_SFLOAT);
            }
            if (!state.hud_depth_pass) {
                state.hud_depth_pass = std::make_unique<OcarinaImagePass<2>>(
                    state.device, state.scheduler, state.pool, OCARINA_HUD_DEPTH_COMP_SPV);
            }
            state.queue.Acquire(state.scheduler, 3);
            state.queue.AddSampledImage(depth, sampler);
            state.queue.AddSampledImage(*state.hud_mask_view, state.upscaler->PointSampler());
            state.queue.AddImage(*resources.frame_gen_depth_view);
            state.hud_depth_pass->Record(state.scheduler, state.queue.UpdateData(),
                                         *resources.frame_gen_depth, extent, false);
            hud_depth = true;
            state.scheduler.InvalidateState();
        }
        ClearHudMotion(state.scheduler, *resources.motion, extent, state.HudZero(),
                       Impl::HUD_ZERO_SIZE);
    }
    AmdFrameGen::Prepare(state.scheduler, {
        .depth = hud_depth ? AmdFrameGen::PrepareImage{*resources.frame_gen_depth,
                                                       VK_FORMAT_R32_SFLOAT,
                                                       VK_IMAGE_ASPECT_COLOR_BIT}
                           : AmdFrameGen::PrepareImage{depth_image, VK_FORMAT_D32_SFLOAT,
                                                       VK_IMAGE_ASPECT_DEPTH_BIT},
        .motion_vectors = {*resources.motion, MOTION_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT},
        .render_size = extent,
        .view_projection = current,
        .reset = reset,
    });
    // A resize can submit the inputs before recording evaluation, so keep the later tick.
    resources.tick = state.scheduler.CurrentTick();
    if (target) {
        state.queue.Acquire(state.scheduler, 2);
        const bool same_size = output_extent.width == target_extent.width &&
                               output_extent.height == target_extent.height;
        const auto output_sampler = same_size ? sampler : state.LinearSampler();
        state.queue.AddSampledImage(state.upscaler->OutputView(), output_sampler);
        state.queue.AddImage(target);
        state.output_pass.RecordOutput(state.scheduler, state.queue.UpdateData(), *state.upscaler,
                                        target_extent, frame, reset);
    }
    state.scheduler.InvalidateState();
    state.last_frame = frame;
    state.last_view_projection = current;
    return true;
}
} // namespace Vulkan
#endif
