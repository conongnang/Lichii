// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

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

} // namespace Vulkan
