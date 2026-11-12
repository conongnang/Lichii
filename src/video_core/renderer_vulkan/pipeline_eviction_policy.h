// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Vulkan {

/// Picks the `count` oldest of a set of eviction candidates by last-use frame. Visit each
/// candidate exactly once, in any order, and evict the ones Take() accepts. Ties at the cutoff
/// age are shared out so that no candidate is kept while a newer one is evicted.
class OldestPipelineSelection {
public:
    /// @param last_uses Last-use frames of every candidate. Reordered in place.
    OldestPipelineSelection(std::span<u64> last_uses, size_t count) {
        count = std::min(count, last_uses.size());
        if (count == 0) {
            return;
        }
        std::ranges::nth_element(last_uses, last_uses.begin() + (count - 1));
        cutoff = last_uses[count - 1];
        const size_t older = std::ranges::count_if(
            last_uses, [this](u64 last_use) { return last_use < cutoff; });
        equal_remaining = count - older;
    }

    /// Whether the candidate last used at `last_use` is evicted.
    bool Take(u64 last_use) {
        if (last_use < cutoff) {
            return true;
        }
        if (last_use == cutoff && equal_remaining != 0) {
            --equal_remaining;
            return true;
        }
        return false;
    }

private:
    u64 cutoff{};
    size_t equal_remaining{};
};

/// Built graphics pipelines kept when the memory query fails (port of zortos293/Lichii#1334).
inline constexpr size_t DefaultKeptPipelines = 4096;

/// Host memory one built graphics pipeline holds, rounded up: building OoT's 12,900 cached
/// pipelines at boot added 3.4 GiB of committed memory (~0.27 MiB each, NVIDIA).
inline constexpr u64 PipelineBytes = 300 * 1024;

/// Built graphics pipelines kept on a machine with `total_memory` bytes of RAM before unused ones
/// are destroyed: as many as fit in a quarter of it (8 GiB: 6,990; 32 GiB: 27,962). A failed
/// memory query (`total_memory` 0) gives DefaultKeptPipelines.
constexpr size_t KeptPipelinesForMemory(u64 total_memory) {
    if (total_memory == 0) {
        return DefaultKeptPipelines;
    }
    return static_cast<size_t>(total_memory / 4 / PipelineBytes);
}

/// Picks which unused pipelines to destroy so that at most `kept` stay built.
/// @param last_uses Frame each candidate was last used (0 for a prebuilt one no draw has used).
/// @param built     Built graphics pipelines in the cache, candidates included.
/// @return Indices into `last_uses`, least recently used first (ties in input order): only as
///         many as it takes to get back to `kept`, or every candidate when that is not enough.
inline std::vector<size_t> PipelinesToEvict(std::span<const u64> last_uses, size_t built,
                                            size_t kept) {
    if (built <= kept) {
        return {};
    }
    std::vector<size_t> order(last_uses.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::stable_sort(order, [&](size_t a, size_t b) { return last_uses[a] < last_uses[b]; });
    order.resize(std::min(built - kept, order.size()));
    return order;
}

} // namespace Vulkan
