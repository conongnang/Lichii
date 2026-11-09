# AMD FidelityFX SDK 2.4.0 (FSR 4 on Vulkan)

Vendored from https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (tag v2.4.0):

- `api/include/` and `upscalers/include/`: the FidelityFX API headers the FSR 4 upscaler
  (`src/video_core/renderer_vulkan/vk_fsr4_upscaler.cpp`) is built against.
- `bin/`: AMD's signed Vulkan DLLs from `Kits/FidelityFX/signedbin`, loaded at runtime on
  Windows and copied next to `lichii.exe` at build time:
  `amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll` (FSR Upscaling 4.1.2).

Redistributed under AMD's license in `LICENSE.md`, which must ship with the binaries.
