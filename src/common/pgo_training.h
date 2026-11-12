// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifdef LICHII_PGO_TRAINING
#include <chrono>
#include <thread>

extern "C" int __llvm_profile_write_file(void);
extern "C" void __llvm_profile_reset_counters(void);
#endif

namespace Common {

/// PGO training builds (-fprofile-generate plus -DLICHII_PGO_TRAINING) write their counters only
/// at a clean exit, and closing a game often hangs until the process is killed. This adds the
/// counts so far to the profile file every 30 s instead (its %m name merges each write).
inline void StartProfileDumps() {
#ifdef LICHII_PGO_TRAINING
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (__llvm_profile_write_file() == 0) {
                __llvm_profile_reset_counters();
            }
        }
    }).detach();
#endif
}

} // namespace Common
