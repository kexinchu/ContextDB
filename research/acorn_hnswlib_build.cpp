#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define HNSWALG_HEADER "hnswalg_acorn.h"
#include "hnswlib.h"
#include "space_l2.h"

using Clock = std::chrono::steady_clock;

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

int main(int argc, char** argv) {
    std::string fbin = "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin";
    std::string out = "research/results/fig1_acorn_iso_m32_1pct/hnswlib_fig1_200k_m32_efc200_acorn.bin";
    int rows = 200000, M = 32, efc = 200;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--fbin") fbin = next();
        else if (a == "--out") out = next();
        else if (a == "--rows") rows = std::stoi(next());
        else if (a == "--m") M = std::stoi(next());
        else if (a == "--ef-construction") efc = std::stoi(next());
        else throw std::runtime_error("unknown arg: " + a);
    }

    int n = 0, d = 0;
    auto xb = read_fbin(fbin, rows, n, d);

    std::cerr << "building hnswlib-ACORN index rows=" << n << " d=" << d
              << " M=" << M << " efc=" << efc << "\n";

    hnswlib::L2Space space(d);
    // Match FAISS ACORN rng seed (12345) for comparable level assignment.
    hnswlib::HierarchicalNSW<float> index(&space, n, M, efc, 12345, false);

    auto t0 = Clock::now();
    for (int i = 0; i < n; i++) {
        index.addPoint(xb.data() + static_cast<size_t>(i) * d, i);
        if ((i + 1) % 10000 == 0) {
            auto elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
            std::cerr << "  added " << (i + 1) << "/" << n << " elapsed_s=" << elapsed << "\n";
        }
    }
    auto build_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cerr << "build done elapsed_s=" << build_s << " maxM0=" << index.maxM0_
              << " M_beta=" << index.acorn_m_beta_ << "\n";

    index.saveIndex(out);
    std::cerr << "saved " << out << "\n";
    return 0;
}
