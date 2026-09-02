#include <faiss/IndexACORN.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

using Clock = std::chrono::steady_clock;

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
            // query_no,query_id,selectivity_pct
            out.push_back({std::stoi(cols[1]), std::stoi(cols[2]), std::stoi(cols[0])});
        } else if (cols.size() == 2) {
            out.push_back({std::stoi(cols[0]), std::stoi(cols[1]), qno++});
        } else {
            continue;
        }
    }
    return out;
}

static std::string labels_to_string(const std::vector<faiss::idx_t>& labels) {
    std::ostringstream out;
    bool first = true;
    for (auto x : labels) {
        if (x < 0) continue;
        if (!first) out << ";";
        out << x;
        first = false;
    }
    return out.str();
}

int main(int argc, char** argv) {
    std::string fbin = "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin";
    std::string workload = "research/results/fig1_200k_q5k_workload.csv";
    std::string out = "research/results/fig1_hnswlib_acorn_q5k.csv";
    int rows = 200000, k = 10, M = 16, efc = 64, gamma = 1;
    std::string ef_arg = "16,32,64,128,256,512,1024,2048,4096";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--fbin") fbin = next();
        else if (a == "--workload-csv") workload = next();
        else if (a == "--out") out = next();
        else if (a == "--rows") rows = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--m") M = std::stoi(next());
        else if (a == "--ef-construction") efc = std::stoi(next());
        else if (a == "--gamma") gamma = std::stoi(next());
        else if (a == "--ef-search-list") ef_arg = next();
        else throw std::runtime_error("unknown arg: " + a);
    }

    int n = 0, d = 0;
    auto xb = read_fbin(fbin, rows, n, d);
    auto requests = read_workload(workload);
    std::vector<int> efs;
    for (const auto& s : split(ef_arg, ',')) efs.push_back(std::stoi(s));
    std::vector<int> metadata(n);
    for (int i = 0; i < n; i++) metadata[i] = i % 100;

    std::cerr << "ACORN-faiss-1 rows=" << n << " d=" << d
              << " requests=" << requests.size() << " efs=" << efs.size() << "\n";

    omp_set_num_threads(1);
    faiss::IndexACORNFlat acorn1(d, M, gamma, metadata, M * 2);
    acorn1.acorn.efConstruction = efc;
    acorn1.add(n, xb.data());

    std::ofstream fout(out);
    fout << "system,ef_search,query_no,query_id,selectivity_pct,k,latency_ms,returned,ids\n";

    std::vector<faiss::idx_t> labels(k, -1);
    std::vector<float> dist(k, 0);
    std::vector<char> filter_map(static_cast<size_t>(n));

    for (int efsv : efs) {
        acorn1.acorn.efSearch = efsv;
        for (const auto& req : requests) {
            if (req.query_id < 0 || req.query_id >= n) {
                throw std::runtime_error("query_id out of range");
            }
            for (int id = 0; id < n; id++) {
                filter_map[id] = (req.selectivity >= 100) || ((id % 100) < req.selectivity);
            }
            std::fill(labels.begin(), labels.end(), -1);
            const float* q = xb.data() + static_cast<size_t>(req.query_id) * d;
            auto t0 = Clock::now();
            acorn1.search(1, q, k, dist.data(), labels.data(), filter_map.data());
            auto t1 = Clock::now();
            int returned = 0;
            for (auto x : labels) if (x >= 0) returned++;
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            fout << "HNSWlib-ACORN," << efsv << "," << req.query_no << ","
                 << req.query_id << "," << req.selectivity << "," << k << ","
                 << ms << "," << returned << "," << labels_to_string(labels) << "\n";
        }
        std::cerr << "finished ef=" << efsv << "\n";
    }
    std::cerr << "wrote " << out << "\n";
    return 0;
}
