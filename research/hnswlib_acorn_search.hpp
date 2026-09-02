#pragma once

// ACORN-1 (gamma=1) filtered search on a loaded hnswlib ACORN index.
// Uses hybrid_greedy_update_nearest on upper levels and hybrid_search_from_candidates on L0.

#include "hnswlib_acorn_hybrid_l0.hpp"

namespace hnswlib_acorn {

template <typename dist_t>
class AcornSearch : public AcornSearchCore<dist_t> {
 public:
    using AcornSearchCore<dist_t>::AcornSearchCore;

    std::vector<hnswlib::labeltype> search(
        const void* query,
        int k,
        int ef_search,
        const std::vector<char>& filter_map) const {
        return this->search_hybrid_l0(query, k, ef_search, filter_map);
    }
};

}  // namespace hnswlib_acorn
