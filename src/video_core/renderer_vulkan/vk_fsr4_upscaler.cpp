// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_fsr4_upscaler.h"
#include "video_core/renderer_vulkan/vk_ffx_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "api/include/ffx_api.h"
#include "api/include/ffx_api_types.h"
#include "api/include/vk/ffx_api_vk.h"
#include "upscalers/include/ffx_upscale.h"
#endif

namespace Vulkan {

namespace {
constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageUsageFlags OUTPUT_USAGE =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
} // Anonymous namespace

#ifdef _WIN32

namespace {

using Api = FfxRuntime::Api;

VkDeviceContext DeviceContext(const Device& device) {
    return VkDeviceContext{
        .vkDevice = *device.GetLogical(),
        .vkPhysicalDevice = static_cast<VkPhysicalDevice>(device.GetPhysical()),
        .vkDeviceProcAddr = device.GetDispatchLoader().vkGetDeviceProcAddr,
    };
}

// The upscaler provider the DLL offers this device: its version id and name.
struct Provider {
    u64 id{};
    std::string name;
};

// FSR 4 when it is wanted and AMD's runtime offers it for this GPU (RX 9000; RX 6000/7000 need
// a recent driver), else FSR 3.1, which runs on any GPU and takes the same inputs: Ocarina's motion
// data, HUD mask and AMD frame generation hang off this upscaler, so they keep working without
// FSR 4. EDEN_NX2_FSR_VERSION=3 forces FSR 3.1 (to test that path on a GPU with FSR 4).
std::optional<Provider> FindFsr4(const Api& api, const Device& device) {
    static const bool force_fsr3 = [] {
        const char* const value = std::getenv("EDEN_NX2_FSR_VERSION");
        return value && value[0] == '3';
    }();
    const bool want_fsr4 = Settings::values.amd_fsr4.GetValue() && !force_fsr3;
    VkDeviceContext vk_context = DeviceContext(device);
    u64 count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = &vk_context;
    versions.outputCount = &count;
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
        LOG_INFO(Render_Vulkan, "FSR 4: the FidelityFX runtime offers no upscaler for this GPU");
        return std::nullopt;
    }
    std::vector<u64> ids(count);
    std::vector<const char*> names(count);
    versions.versionIds = ids.data();
    versions.versionNames = names.data();
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK) {
        return std::nullopt;
    }
    std::string offered;
    std::optional<Provider> fsr4;
    std::optional<Provider> fsr3;
    for (u64 i = 0; i < count; ++i) {
        const std::string name = names[i] ? names[i] : "";
        offered += (offered.empty() ? "" : ", ") + name;
        // The list is newest first; take the newest 4.x and the newest 3.1.x.
        if (!fsr4 && !name.empty() && name[0] >= '4' && name[0] <= '9') {
            fsr4 = Provider{ids[i], name};
        }
        if (!fsr3 && name.starts_with("3.") && name.size() > 2 && name[2] >= '1') {
            fsr3 = Provider{ids[i], name};
        }
    }
    LOG_INFO(Render_Vulkan, "AMD FSR: upscalers offered for {}: {}", device.GetModelName(),
             offered);
    if (want_fsr4 && fsr4) {
        return fsr4;
    }
    if (fsr3) {
        LOG_INFO(Render_Vulkan, "AMD FSR: using FSR {} ({})", fsr3->name,
                 force_fsr3                                 ? "forced by EDEN_NX2_FSR_VERSION=3"
                 : !Settings::values.amd_fsr4.GetValue()    ? "FSR 4 is turned off"
                                                            : "no FSR 4 for this GPU or driver");
    }
    return fsr3;
}

FfxApiResource ImageResource(const Fsr4Upscaler::Image& image, VkExtent2D size, u32 state,
                             u32 usage) {
    FfxApiResourceDescription description{};
    description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    description.format = ffxApiGetSurfaceFormatVK(image.format);
    description.width = size.width;
    description.height = size.height;
    description.depth = 1;
    description.mipCount = 1;
    description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    description.usage = usage;
    return ffxApiGetResourceVK(reinterpret_cast<void*>(image.image), description, state);
}

VkImageMemoryBarrier Barrier(const Fsr4Upscaler::Image& image, VkAccessFlags src, VkAccessFlags dst,
                             VkImageLayout old_layout, VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

} // Anonymous namespace

struct Fsr4Upscaler::Context {
    std::shared_ptr<const Api> runtime; ///< Released after the context (member order)
    const Api* api{};
    ffxContext handle{};
    Provider provider;
};

namespace {
struct ProbeResult {
    bool available{};
    bool fsr4{};
};
ProbeResult Probe(const Device& device) {
    static std::once_flag once;
    static ProbeResult result{};
    std::call_once(once, [&device] {
        const auto runtime = FfxRuntime::Acquire();
        const auto provider = runtime ? FindFsr4(*runtime, device) : std::nullopt;
        result.available = provider.has_value();
        result.fsr4 = provider && !provider->name.empty() && provider->name[0] >= '4';
    });
    return result;
}
} // Anonymous namespace

bool Fsr4Upscaler::IsAvailable(const Device& device) {
    // AMD frame generation needs this upscaler's motion data even with FSR 4 turned off.
    if (!Settings::values.amd_fsr4.GetValue()) {
        return false;
    }
    return Probe(device).available;
}

bool Fsr4Upscaler::IsFsr4(const Device& device) {
    return IsAvailable(device) && Probe(device).fsr4;
}

Fsr4Upscaler::Fsr4Upscaler(const Device& device_, Scheduler& scheduler_,
                           MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_},
      context{std::make_unique<Context>()} {
    context->runtime = FfxRuntime::Acquire();
    context->api = context->runtime.get();
    if (!context->api) {
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    const auto provider = FindFsr4(*context->api, device);
    if (!provider) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    context->provider = *provider;
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

Fsr4Upscaler::~Fsr4Upscaler() {
    scheduler.Finish();
    DestroyContext();
}

void Fsr4Upscaler::DestroyContext() {
    if (context && context->handle) {
        context->api->destroy_context(&context->handle, nullptr);
        context->handle = nullptr;
    }
}

bool Fsr4Upscaler::CreateContext(VkExtent2D render_size, VkExtent2D upscale_size,
                                 bool depth_inverted, bool hdr_requested) {
    DestroyContext();
    // EDEN_NX2_FSR4_HDR=0 treats the colour as tonemapped (non-linear) for A/B runs.
    static const bool hdr_allowed = [] {
        const char* const value = std::getenv("EDEN_NX2_FSR4_HDR");
        return !value || std::string_view{value} != "0";
    }();
    const bool hdr = hdr_requested && hdr_allowed;
    VkDeviceContext vk_context = DeviceContext(device);
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.deviceContext = vk_context;
    ffxOverrideVersion version_override{};
    version_override.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version_override.versionId = context->provider.id;
    ffxCreateContextDescUpscaleVersion api_version{};
    api_version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    api_version.version = FFX_UPSCALER_VERSION;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags = (depth_inverted ? FFX_UPSCALE_ENABLE_DEPTH_INVERTED : 0u) |
                   (hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE
                        : FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE);
    create.maxRenderSize = {render_size.width, render_size.height};
    create.maxUpscaleSize = {upscale_size.width, upscale_size.height};
    create.fpMessage = nullptr;
    // Chain: upscale <- version <- override <- backend.
    create.header.pNext = &api_version.header;
    api_version.header.pNext = &version_override.header;
    version_override.header.pNext = &backend.header;
    const ffxReturnCode_t result =
        context->api->create_context(&context->handle, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "AMD FSR {}: context creation failed ({}), using the game's upscaler",
                  context->provider.name, result);
        context->handle = nullptr;
        return false;
    }
    ffxQueryGetProviderVersion provider{};
    provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    context->api->query(&context->handle, &provider.header);
    LOG_INFO(Render_Vulkan, "AMD FSR: upscaler {} running, {}x{} -> {}x{} ({})",
             provider.versionName ? provider.versionName : context->provider.name.c_str(),
             render_size.width, render_size.height, upscale_size.width, upscale_size.height,
             hdr ? "HDR" : "non-linear colour");
    return true;
}

void Fsr4Upscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

bool Fsr4Upscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    const bool extent_changed = !output_image || in.render_size.width != render_extent.width ||
                                in.render_size.height != render_extent.height ||
                                in.upscale_size.width != upscale_extent.width ||
                                in.upscale_size.height != upscale_extent.height;
    const bool options_changed = extent_changed || in.depth_inverted != depth_inverted_input ||
                                 in.hdr != hdr_input;
    if (options_changed) {
        scheduler.Finish(); // the old context may still be recording
        if (extent_changed) {
            CreateOutput(in.upscale_size);
        }
        if (!CreateContext(in.render_size, in.upscale_size, in.depth_inverted, in.hdr)) {
            failed = true;
            return false;
        }
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
        depth_inverted_input = in.depth_inverted;
        hdr_input = in.hdr;
    }
    const bool reset = in.reset || options_changed;

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, output = *output_image](vk::CommandBuffer cmdbuf) {
        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // FSR reads them as sampled images (SHADER_READ_ONLY_OPTIMAL, as AMD's docs ask).
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);

        const Image output_image_info{
            .view = VK_NULL_HANDLE,
            .image = output,
            .format = OUTPUT_FORMAT,
            .range{},
            .barrier_aspect = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        ffxDispatchDescUpscale dispatch{};
        dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dispatch.commandList = reinterpret_cast<void*>(*cmdbuf);
        dispatch.color = ImageResource(in.color, in.render_size,
                                       FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                       FFX_API_RESOURCE_USAGE_READ_ONLY);
        dispatch.depth = ImageResource(in.depth, in.render_size,
                                       FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                       FFX_API_RESOURCE_USAGE_READ_ONLY |
                                           FFX_API_RESOURCE_USAGE_DEPTHTARGET);
        dispatch.motionVectors = ImageResource(in.motion_vectors, in.render_size,
                                               FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                               FFX_API_RESOURCE_USAGE_READ_ONLY);
        dispatch.exposure = {};
        dispatch.reactive = {};
        dispatch.transparencyAndComposition = {};
        dispatch.output = ImageResource(output_image_info, in.upscale_size,
                                        FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
                                        FFX_API_RESOURCE_USAGE_UAV);
        dispatch.jitterOffset = {in.jitter[0], in.jitter[1]};
        dispatch.motionVectorScale = {in.motion_vector_scale[0], in.motion_vector_scale[1]};
        dispatch.renderSize = {in.render_size.width, in.render_size.height};
        dispatch.upscaleSize = {in.upscale_size.width, in.upscale_size.height};
        dispatch.enableSharpening = in.sharpen;
        dispatch.sharpness = in.sharpness;
        dispatch.frameTimeDelta = in.frame_time_ms;
        dispatch.preExposure = 1.0f;
        dispatch.reset = reset;
        // Only frame generation reads the camera; FSR 4 upscaling ignores these.
        dispatch.cameraNear = 0.1f;
        dispatch.cameraFar = 1000.0f;
        dispatch.cameraFovAngleVertical = 1.0f;
        dispatch.viewSpaceToMetersFactor = 1.0f;
        dispatch.flags = 0;
        const ffxReturnCode_t result = context->api->dispatch(&context->handle, &dispatch.header);

        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                               to_general);
        if (result != FFX_API_RETURN_OK && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan, "FSR 4: dispatch failed ({}), using FSR 3.1 from the next "
                                     "frame",
                      result);
        }
    });
    return true;
}

#else // _WIN32

struct Fsr4Upscaler::Context {};

bool Fsr4Upscaler::IsAvailable(const Device&) {
    return false; // AMD ships the FSR 4 runtime for Windows only
}

bool Fsr4Upscaler::IsFsr4(const Device&) {
    return false;
}

Fsr4Upscaler::Fsr4Upscaler(const Device& device_, Scheduler& scheduler_,
                           MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
}

Fsr4Upscaler::~Fsr4Upscaler() = default;

void Fsr4Upscaler::CreateOutput(VkExtent2D) {}

bool Fsr4Upscaler::CreateContext(VkExtent2D, VkExtent2D, bool, bool) {
    return false;
}

void Fsr4Upscaler::DestroyContext() {}

bool Fsr4Upscaler::Upscale(const Inputs&) {
    return false;
}

#endif // _WIN32

} // namespace Vulkan
