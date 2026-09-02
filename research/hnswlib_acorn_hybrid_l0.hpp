#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "hnswalg.h"

namespace hnswlib_acorn {

struct MinimaxHeap {
    int n = 0;
    int k = 0;
    int nvalid = 0;
    std::vector<int> ids;
    std::vector<float> dis;

    explicit MinimaxHeap(int capacity) : n(capacity), ids(capacity, -1), dis(capacity, 0.0f) {}

    void push(int id, float dist) {
        if (k == n) {
            if (dist >= dis[0]) return;
            remove_max();
        }
        dis[k] = dist;
        ids[k] = id;
        ++k;
        ++nvalid;
        sift_up(k - 1);
    }

    int size() const { return nvalid; }

    int pop_min(float* vmin_out = nullptr) {
        int imin = -1;
        float vmin = 0.0f;
        for (int i = 0; i < k; ++i) {
            if (ids[i] == -1) continue;
            if (imin == -1 || dis[i] < vmin) {
                imin = i;
                vmin = dis[i];
            }
        }
        if (imin == -1) return -1;
        if (vmin_out) *vmin_out = vmin;
        int ret = ids[imin];
        ids[imin] = -1;
        --nvalid;
        return ret;
    }

    int count_below(float thresh) const {
        int below = 0;
        for (int i = 0; i < k; ++i) {
            if (dis[i] < thresh) ++below;
        }
        return below;
    }

 private:
    void sift_up(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (dis[parent] >= dis[idx]) break;
            std::swap(dis[parent], dis[idx]);
            std::swap(ids[parent], ids[idx]);
            idx = parent;
        }
    }

    void remove_max() {
        if (k <= 0) return;
        --k;
        --nvalid;
        if (k == 0) return;
        dis[0] = dis[k];
        ids[0] = ids[k];
        ids[k] = -1;
        int idx = 0;
        while (true) {
            int left = 2 * idx + 1;
            int right = left + 1;
            int largest = idx;
            if (left < k && dis[left] > dis[largest]) largest = left;
            if (right < k && dis[right] > dis[largest]) largest = right;
            if (largest == idx) break;
            std::swap(dis[idx], dis[largest]);
            std::swap(ids[idx], ids[largest]);
            idx = largest;
        }
    }
};

struct ResultHeap {
    int k = 0;
    std::vector<std::pair<float, hnswlib::labeltype>> items;

    explicit ResultHeap(int k_) : k(k_) {}

    void push(float dist, hnswlib::labeltype label) {
        if ((int)items.size() < k) {
            items.emplace_back(dist, label);
            std::push_heap(items.begin(), items.end(),
                           [](const auto& a, const auto& b) { return a.first < b.first; });
            return;
        }
        if (dist >= items.front().first) return;
        std::pop_heap(items.begin(), items.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        items.back() = {dist, label};
        std::push_heap(items.begin(), items.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    std::vector<hnswlib::labeltype> sorted_labels() const {
        auto copy = items;
        std::sort(copy.begin(), copy.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<hnswlib::labeltype> out;
        for (const auto& item : copy) out.push_back(item.second);
        return out;
    }
};

struct HybridL0Stats {
    int pops = 0;
    int num_found_breaks = 0;
    int ef_breaks = 0;
    int filtered_seen = 0;
    int filtered_seen_visited = 0;
    int filtered_new = 0;
    int max_num_found_at_break = 0;
    int max_neighbors_scanned_one_pop = 0;
};

template <typename dist_t>
class AcornSearchCore {
 public:
    explicit AcornSearchCore(hnswlib::HierarchicalNSW<dist_t>* index, int m_beta_multiplier = 2)
        : index_(index), m_beta_((int)index->M_ * m_beta_multiplier) {}

    dist_t dist(const void* query, hnswlib::tableint id) const {
        if (index_->isMarkedDeleted(id)) return std::numeric_limits<dist_t>::max();
        return index_->fstdistfunc_(query, index_->getDataByInternalId(id), index_->dist_func_param_);
    }

    void load_neighbors(hnswlib::tableint id, int level, std::vector<hnswlib::tableint>& out) const {
        out.clear();
        if (level == 0) {
            auto* ll = index_->get_linklist0(id);
            size_t sz = index_->getListCount(ll);
            auto* data = (hnswlib::tableint*)(ll + 1);
            out.assign(data, data + sz);
            return;
        }
        auto* ll = (unsigned int*)index_->get_linklist(id, level);
        int sz = index_->getListCount(ll);
        auto* data = (hnswlib::tableint*)(ll + 1);
        out.assign(data, data + sz);
    }

    void hybrid_greedy_update_nearest(const void* query, const std::vector<char>& filter_map, int level,
                                      hnswlib::tableint& nearest, dist_t& d_nearest) const {
        const int m = (int)index_->M_;
        std::vector<hnswlib::tableint> n1, n2;
        for (;;) {
            hnswlib::tableint prev = nearest;
            load_neighbors(nearest, level, n1);
            int num_found = 0;
            for (hnswlib::tableint v : n1) {
                if (filter_map[v]) ++num_found;
                else if (gamma_ > 1) continue;
                if (filter_map[v]) {
                    dist_t d = dist(query, v);
                    if (d < d_nearest || !filter_map[nearest]) {
                        nearest = v;
                        d_nearest = d;
                    }
                    if (num_found >= m) break;
                }
                if (gamma_ == 1) {
                    load_neighbors(v, level, n2);
                    for (hnswlib::tableint v2 : n2) {
                        if (!filter_map[v2]) continue;
                        ++num_found;
                        dist_t d2 = dist(query, v2);
                        if (d2 < d_nearest || !filter_map[nearest]) {
                            nearest = v2;
                            d_nearest = d2;
                        }
                        if (num_found >= m) break;
                    }
                }
            }
            if (nearest == prev) return;
        }
    }

    std::vector<hnswlib::labeltype> search_hybrid_l0(
        const void* query, int k, int ef_search, const std::vector<char>& filter_map,
        HybridL0Stats* stats = nullptr, bool count_new_only = false) const {
        if (index_->cur_element_count == 0) return {};

        hnswlib::tableint nearest = index_->enterpoint_node_;
        dist_t d_nearest = dist(query, nearest);
        for (int level = index_->maxlevel_; level >= 1; --level) {
            hybrid_greedy_update_nearest(query, filter_map, level, nearest, d_nearest);
        }

        const int ef = std::max(ef_search, k);
        MinimaxHeap candidates(ef);
        candidates.push((int)nearest, (float)d_nearest);
        ResultHeap results(k);

        auto* vl = index_->visited_list_pool_->getFreeVisitedList();
        vl->reset();
        auto* visited = vl->mass;
        auto tag = vl->curV;

        hybrid_search_from_candidates(query, filter_map, candidates, results, visited, tag, ef, stats,
                                      count_new_only);

        index_->visited_list_pool_->releaseVisitedList(vl);
        return results.sorted_labels();
    }

    std::vector<hnswlib::labeltype> search_bfs_l0(
        const void* query, int k, int ef_search, const std::vector<char>& filter_map,
        hnswlib::tableint nearest_seed, dist_t d_seed) const {
        struct F : hnswlib::BaseFilterFunctor {
            const std::vector<char>* map;
            explicit F(const std::vector<char>* m) : map(m) {}
            bool operator()(hnswlib::labeltype id) override {
                return id >= 0 && static_cast<size_t>(id) < map->size() && (*map)[id];
            }
        } filt(&filter_map);
        const int ef = std::max(ef_search, k);
        auto pq = index_->template searchBaseLayerST<false>(nearest_seed, query, ef, &filt);
        std::vector<std::pair<float, hnswlib::tableint>> items;
        while (!pq.empty()) {
            items.emplace_back(pq.top().first, pq.top().second);
            pq.pop();
        }
        std::sort(items.begin(), items.end());
        std::vector<hnswlib::labeltype> out;
        for (int i = 0; i < k && i < (int)items.size(); ++i) {
            out.push_back(index_->getExternalLabel(items[i].second));
        }
        return out;
    }

 private:
    hnswlib::HierarchicalNSW<dist_t>* index_;
    int m_beta_;
    int gamma_{1};

    void hybrid_search_from_candidates(
        const void* query, const std::vector<char>& filter_map, MinimaxHeap& candidates,
        ResultHeap& results, hnswlib::vl_type* visited, hnswlib::vl_type tag, int ef_search,
        HybridL0Stats* stats, bool count_new_only) const {
        const int m2 = (int)index_->M_ * 2;

        for (int i = 0; i < candidates.size(); ++i) {
            hnswlib::tableint v1 = (hnswlib::tableint)candidates.ids[i];
            results.push(candidates.dis[i], index_->getExternalLabel(v1));
            visited[v1] = tag;
        }

        std::vector<hnswlib::tableint> n1, n2;
        while (candidates.size() > 0) {
            float d0 = 0.0f;
            int v0 = candidates.pop_min(&d0);
            if (v0 < 0) break;
            if (candidates.count_below(d0) >= ef_search) {
                if (stats) ++stats->ef_breaks;
                break;
            }
            if (stats) ++stats->pops;

            hnswlib::tableint node0 = (hnswlib::tableint)v0;
            int num_found = 0;
            bool keep_expanding = true;
            int scanned = 0;

            load_neighbors(node0, 0, n1);
            for (size_t j = 0; j < n1.size(); ++j) {
                ++scanned;
                hnswlib::tableint v1 = n1[j];
                if (filter_map[v1]) {
                    if (!count_new_only) ++num_found;
                    if (stats) {
                        ++stats->filtered_seen;
                        if (visited[v1] == tag) ++stats->filtered_seen_visited;
                    }
                }
                if (visited[v1] == tag) continue;
                if (filter_map[v1]) {
                    visited[v1] = tag;
                    dist_t d1 = dist(query, v1);
                    results.push((float)d1, index_->getExternalLabel(v1));
                    candidates.push((int)v1, (float)d1);
                    if (count_new_only) ++num_found;
                    if (stats) ++stats->filtered_new;
                    if (num_found >= m2) {
                        keep_expanding = false;
                        if (stats) {
                            ++stats->num_found_breaks;
                            stats->max_num_found_at_break = std::max(stats->max_num_found_at_break, num_found);
                        }
                        break;
                    }
                }

                if (((j >= (size_t)m_beta_) && keep_expanding) || gamma_ == 1) {
                    load_neighbors(v1, 0, n2);
                    scanned += (int)n2.size();
                    for (hnswlib::tableint v2 : n2) {
                        if (filter_map[v2]) {
                            if (!count_new_only) ++num_found;
                            if (stats) {
                                ++stats->filtered_seen;
                                if (visited[v2] == tag) ++stats->filtered_seen_visited;
                            }
                        } else {
                            continue;
                        }
                        if (visited[v2] == tag) continue;
                        visited[v2] = tag;
                        dist_t d2 = dist(query, v2);
                        results.push((float)d2, index_->getExternalLabel(v2));
                        candidates.push((int)v2, (float)d2);
                        if (count_new_only) ++num_found;
                        if (stats) ++stats->filtered_new;
                        if (num_found >= m2) {
                            keep_expanding = false;
                            if (stats) {
                                ++stats->num_found_breaks;
                                stats->max_num_found_at_break = std::max(stats->max_num_found_at_break, num_found);
                            }
                            break;
                        }
                    }
                }
            }
            if (stats) {
                stats->max_neighbors_scanned_one_pop =
                    std::max(stats->max_neighbors_scanned_one_pop, scanned);
            }
        }
    }
};

}  // namespace hnswlib_acorn
