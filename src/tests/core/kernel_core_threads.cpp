// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/fiber.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "tests/core/system_test_context.h"

// Each core's scheduler only points at its main and idle threads. Kernel shutdown left them open,
// so they went with the slab heaps unfinalized and their fibers (2 MiB stacks each) leaked, 8 per
// kernel.
TEST_CASE("Kernel shutdown releases the cores' main and idle thread fibers", "[core][kernel]") {
    const bool multicore = GENERATE(false, true);
    INFO("multicore: " << multicore);
    ScopedSystemTestContext environment;
    Settings::values.use_multi_core.SetValue(multicore);
    Core::System system;
    system.Initialize();
    auto& kernel = system.Kernel();
    std::optional<Kernel::KThread> thread{std::in_place, kernel};
    REQUIRE(Kernel::KThread::InitializeDummyThread(system, &*thread, nullptr).IsSuccess());
    kernel.SetCurrentEmuThread(&*thread);
    kernel.Initialize();
    REQUIRE(kernel.IsMulticore() == multicore);

    std::vector<std::weak_ptr<Common::Fiber>> fibers;
    for (std::size_t core = 0; core < Core::Hardware::NUM_CPU_CORES; ++core) {
        auto& scheduler = kernel.Scheduler(core);
        // Before any core runs, the current thread is the main thread.
        for (auto* each : {scheduler.GetSchedulerCurrentThread(), scheduler.GetIdleThread()}) {
            REQUIRE(each != nullptr);
            REQUIRE(each->GetHostContext() != nullptr);
            fibers.emplace_back(each->GetHostContext());
        }
    }
    REQUIRE(fibers.size() == 2 * Core::Hardware::NUM_CPU_CORES);

    system.CoreTiming().ClearPendingEvents();
    kernel.Shutdown();
    for (const auto& fiber : fibers) {
        CHECK(fiber.expired());
    }
    kernel.SetCurrentEmuThread(nullptr);
    kernel.UnregisterKernelObject(&*thread);
    thread.reset();
}
