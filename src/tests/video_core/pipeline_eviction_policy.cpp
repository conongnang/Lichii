// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/pipeline_eviction_policy.h"

namespace {

void CheckOldest(std::span<const u64> last_uses, size_t count) {
    std::vector<u64> candidates(last_uses.begin(), last_uses.end());
    Vulkan::OldestPipelineSelection selection{candidates, count};
    std::vector<u64> evicted;
    for (const u64 last_use : last_uses) {
        if (evicted.size() == count) {
            break;
        }
        if (selection.Take(last_use)) {
            evicted.push_back(last_use);
        }
    }
    std::vector<u64> expected(last_uses.begin(), last_uses.end());
    std::ranges::sort(expected);
    expected.resize(std::min(count, expected.size()));
    std::ranges::sort(evicted);
    CHECK(evicted == expected);
}

} // namespace

TEST_CASE("Pipeline eviction preserves strict age ordering across cutoff ties",
          "[video_core][pipeline_eviction]") {
    std::array<u64, 3> last_uses{100, 200, 200};
    do {
        CAPTURE(last_uses);
        CheckOldest(last_uses, 2);
    } while (std::ranges::next_permutation(last_uses).found);
}

TEST_CASE("Pipeline eviction selects the requested number of oldest candidates",
          "[video_core][pipeline_eviction]") {
    const std::array<u64, 0> empty{};
    CheckOldest(empty, 0);
    CheckOldest(empty, 1);
    const std::array<u64, 4> mixed{300, 100, 200, 100};
    const std::array<u64, 4> equal{0, 0, 0, 0};
    for (size_t count = 0; count <= mixed.size() + 1; ++count) {
        CAPTURE(count);
        CheckOldest(mixed, count);
        CheckOldest(equal, count);
    }
}

TEST_CASE("Pipeline eviction handles the largest last-use frame without overflow",
          "[video_core][pipeline_eviction]") {
    constexpr u64 maximum = std::numeric_limits<u64>::max();
    const std::array<u64, 3> last_uses{maximum, maximum, maximum - 1};
    for (size_t count = 1; count <= last_uses.size(); ++count) {
        CAPTURE(count);
        CheckOldest(last_uses, count);
    }
}
