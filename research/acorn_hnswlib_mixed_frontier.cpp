#include "hnswlib_acorn_search.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hnswlib.h"
#include "space_l2.h"

using Clock = std::chrono::steady_clock;

struct IdFilter : hnswlib::BaseFilterFunctor {
    const std::vector<char>* map;
    explicit IdFilter(const std::vector<char>* m) : map(m) {}
    bool operator()(hnswlib::labeltype id) override {
        return id >= 0 && static_cast<size_t>(id) < map->size() && (*map)[id];
    }
};

struct Request {
    int query_id;
    int selectivity;
    int query_no;
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static std::vector<float> read_fbin(const std::string& path, int rows, int& n, int& d) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open fbin: " + path);
    int32_t file_n = 0, file_d = 0;
    in.read(reinterpret_cast<char*>(&file_n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&file_d), sizeof(int32_t));
    n = std::min(rows, file_n);
    d = file_d;
    std::vector<float> xb(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(xb.data()), xb.size() * sizeof(float));
    if (!in) throw std::runtime_error("failed reading fbin payload");
    return xb;
}

static std::vector<Request> read_workload(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open workload: " + path);
    std::string line;
    std::getline(in, line);
    std::vector<Request> out;
    int qno = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = split(line, ',');
        if (cols.size() >= 3) {
            out.push_back({std::stoi(cols[1]), std::stoi(cols[2]), std::stoi(cols[0])});
        } else if (cols.size() == 2) {
            out.push_back({std::stoi(cols[0]), std::stoi(cols[1]), qno++});
        }
    }
    return out;
}

static std::string labels_to_string(const std::vector<hnswlib::labeltype>& labels) {
    std::ostringstream out;
    bool first = true;
    for (auto x : labels) {
        if (!first) out << ";";
        out << x;
        first = false;
    }
    return out.str();
}

int main(int argc, char** argv) {
    std::string fbin = "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin";
    std::string workload = "research/results/fig1_200k_q5k_workload.csv";
    std::string index_path = "research/results/fig1_four_curve_m32_5pct/hnswlib_fig1_200k_m32_efc200.bin";
    std::string out = "research/results/hnswlib_acorn_raw.csv";
    int rows = 200000, k = 10;
    std::string mode = "acorn";
    std::string ef_arg = "16,32,64,128,256,512,1024,2048,4096,8192";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--fbin") fbin = next();
        else if (a == "--workload-csv") workload = next();
        else if (a == "--index") index_path = next();
        else if (a == "--out") out = next();
        else if (a == "--rows") rows = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--mode") mode = next();
        else if (a == "--ef-search-list") ef_arg = next();
        else throw std::runtime_error("unknown arg: " + a);
    }

    int n = 0, d = 0;
    auto xb = read_fbin(fbin, rows, n, d);
    auto requests = read_workload(workload);
    std::vector<int> efs;
    for (const auto& s : split(ef_arg, ',')) efs.push_back(std::stoi(s));

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, n, true);
    std::unique_ptr<hnswlib_acorn::AcornSearch<float>> acorn;
    if (mode == "acorn") {
        acorn = std::make_unique<hnswlib_acorn::AcornSearch<float>>(&index, 2);
    }

    std::cerr << "hnswlib-" << mode << " rows=" << n << " d=" << d << " M=" << index.M_
              << " maxM0=" << index.maxM0_ << " requests=" << requests.size() << " efs=" << efs.size() << "\n";

    std::ofstream fout(out);
    fout << "system,ef_search,query_no,query_id,selectivity_pct,k,latency_ms,returned,ids\n";

    std::vector<char> filter_map(n, 0);
    for (int efsv : efs) {
        index.setEf(efsv);
        IdFilter filt(&filter_map);
        for (const auto& req : requests) {
            if (req.query_id < 0 || req.query_id >= n) {
                throw std::runtime_error("query_id out of range");
            }
            for (int id = 0; id < n; id++) {
                filter_map[id] = (req.selectivity >= 100) || ((id % 100) < req.selectivity);
            }
            const float* q = xb.data() + static_cast<size_t>(req.query_id) * d;
            auto t0 = Clock::now();
            std::vector<hnswlib::labeltype> labels;
            if (mode == "acorn") {
                labels = acorn->search(q, k, efsv, filter_map);
            } else {
                auto res = index.searchKnn(q, k, &filt);
                while (!res.empty()) {
                    labels.push_back(res.top().second);
                    res.pop();
                }
            }
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const char* system_name = (mode == "acorn") ? "HNSWlib-ACORN-native" : "HNSWlib-sweeping";
            fout << system_name << "," << efsv << "," << req.query_no << ","
                 << req.query_id << "," << req.selectivity << "," << k << ","
                 << ms << "," << labels.size() << "," << labels_to_string(labels) << "\n";
        }
        std::cerr << "finished ef=" << efsv << "\n";
    }
    std::cerr << "wrote " << out << "\n";
    return 0;
}
