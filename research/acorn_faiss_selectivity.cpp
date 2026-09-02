#include <faiss/IndexACORN.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/IDSelector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>

using Clock = std::chrono::steady_clock;

struct Result {
    std::string system;
    int repeat;
    int selectivity;
    int query_no;
    int query_id;
    int k;
    int overfetch;
    double latency_ms;
    int returned;
    std::string ids;
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

static std::vector<int> read_query_ids(const std::string& path, int limit) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open query csv: " + path);
    std::string line;
    std::getline(in, line);
    std::vector<int> ids;
    while (std::getline(in, line) && static_cast<int>(ids.size()) < limit) {
        if (!line.empty()) ids.push_back(std::stoi(line));
    }
    return ids;
}

static void write_detail(const std::string& path, const std::vector<Result>& rows) {
    std::ofstream out(path);
    out << "system,repeat,selectivity_pct,query_no,query_id,k,overfetch,latency_ms,returned,ids\n";
    for (const auto& r : rows) {
        out << r.system << "," << r.repeat << "," << r.selectivity << "," << r.query_no << ","
            << r.query_id << "," << r.k << "," << r.overfetch << ","
            << r.latency_ms << "," << r.returned << "," << r.ids << "\n";
    }
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

static double mean(std::vector<double> v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double p95(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = std::max<size_t>(0, static_cast<size_t>(std::ceil(0.95 * v.size())) - 1);
    return v[std::min(idx, v.size() - 1)];
}

static void write_summary(const std::string& path, const std::vector<Result>& rows) {
    std::map<std::pair<std::string, int>, std::vector<const Result*>> groups;
    for (const auto& r : rows) groups[{r.system, r.selectivity}].push_back(&r);
    std::ofstream out(path);
    out << "system,selectivity_pct,repeats,queries,latency_ms_mean,latency_ms_p50,latency_ms_p95,returned_mean,full_k_rate\n";
    for (const auto& [key, items] : groups) {
        std::vector<double> lat, ret;
        std::vector<int> repeats;
        int full_k = 0;
        int k = items.front()->k;
        for (const auto* r : items) {
            lat.push_back(r->latency_ms);
            ret.push_back(r->returned);
            repeats.push_back(r->repeat);
            if (r->returned >= k) full_k++;
        }
        std::sort(repeats.begin(), repeats.end());
        repeats.erase(std::unique(repeats.begin(), repeats.end()), repeats.end());
        out << key.first << "," << key.second << "," << repeats.size() << ","
            << items.size() << "," << mean(lat) << "," << median(lat) << "," << p95(lat) << ","
            << mean(ret) << "," << (static_cast<double>(full_k) / items.size()) << "\n";
    }
}

static bool includes_system(const std::vector<std::string>& systems, const std::string& name) {
    return std::find(systems.begin(), systems.end(), name) != systems.end();
}

int main(int argc, char** argv) {
    std::string fbin = "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin";
    std::string query_csv = "research/results/amazon_200k_query_ids_100.csv";
    std::string out = "research/results/acorn_faiss_200k_q100.csv";
    int rows = 200000, queries = 100, k = 10, M = 16, efc = 64, efs = 128, gamma = 12, repeats = 1;
    std::string selectivity_arg = "1,2,5,10,20,30,40,50,60,70,80,90,100";
    std::string systems_arg = "ACORN-faiss-gamma,ACORN-faiss-1,FAISS-HNSW-sweeping";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--fbin") fbin = next();
        else if (a == "--query-id-csv") query_csv = next();
        else if (a == "--out") out = next();
        else if (a == "--rows") rows = std::stoi(next());
        else if (a == "--queries") queries = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--m") M = std::stoi(next());
        else if (a == "--ef-construction") efc = std::stoi(next());
        else if (a == "--ef-search") efs = std::stoi(next());
        else if (a == "--gamma") gamma = std::stoi(next());
        else if (a == "--repeats") repeats = std::stoi(next());
        else if (a == "--systems") systems_arg = next();
        else if (a == "--selectivities") selectivity_arg = next();
        else throw std::runtime_error("unknown arg: " + a);
    }

    int n = 0, d = 0;
    auto xb = read_fbin(fbin, rows, n, d);
    auto query_ids = read_query_ids(query_csv, queries);
    std::vector<int> sels;
    for (const auto& s : split(selectivity_arg, ',')) sels.push_back(std::stoi(s));
    std::vector<std::string> systems = split(systems_arg, ',');
    std::vector<int> metadata(n);
    for (int i = 0; i < n; i++) metadata[i] = i % 100;

    std::cerr << "rows=" << n << " d=" << d << " queries=" << query_ids.size()
              << " M=" << M << " efc=" << efc << " efs=" << efs << "\n";

    omp_set_num_threads(1);
    faiss::IndexHNSWFlat hnsw(d, M, 1);
    hnsw.hnsw.efConstruction = efc;
    hnsw.hnsw.efSearch = efs;
    hnsw.add(n, xb.data());

    faiss::IndexACORNFlat acorn(d, M, gamma, metadata, M * 2);
    acorn.acorn.efConstruction = efc;
    acorn.acorn.efSearch = efs;
    acorn.add(n, xb.data());

    faiss::IndexACORNFlat acorn1(d, M, 1, metadata, M * 2);
    acorn1.acorn.efConstruction = efc;
    acorn1.acorn.efSearch = efs;
    acorn1.add(n, xb.data());

    std::vector<Result> results;
    std::vector<faiss::idx_t> labels;
    std::vector<float> dist;
    for (int sel : sels) {
        int overfetch = std::min(n, std::max(k, static_cast<int>(std::ceil(k * 4.0 * 100.0 / std::max(sel, 1)))));
        std::vector<uint8_t> selector_bitmap((static_cast<size_t>(n) + 7) / 8);
        for (int id = 0; id < n; id++) {
            if (sel >= 100 || ((id % 100) < sel)) {
                selector_bitmap[static_cast<size_t>(id) / 8] |= static_cast<uint8_t>(1u << (id % 8));
            }
        }
        faiss::IDSelectorBitmap selector(selector_bitmap.size(), selector_bitmap.data());
        faiss::SearchParametersHNSW hnsw_params;
        hnsw_params.efSearch = efs;
        hnsw_params.sel = &selector;
        std::vector<char> filter_map(static_cast<size_t>(query_ids.size()) * n);
        for (size_t qi = 0; qi < query_ids.size(); qi++) {
            for (int id = 0; id < n; id++) {
                filter_map[qi * n + id] = (sel >= 100) || ((id % 100) < sel);
            }
        }

        for (size_t qi = 0; qi < query_ids.size(); qi++) {
            int qid = query_ids[qi];
            const float* q = xb.data() + static_cast<size_t>(qid) * d;

            auto run_acorn = [&](const std::string& name, faiss::IndexACORNFlat& idx, int repeat) {
                labels.assign(k, -1);
                dist.assign(k, 0);
                auto t0 = Clock::now();
                idx.search(1, q, k, dist.data(), labels.data(), filter_map.data() + qi * n);
                auto t1 = Clock::now();
                int returned = 0;
                for (auto x : labels) if (x >= 0) returned++;
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                results.push_back({name, repeat, sel, static_cast<int>(qi), qid, k, overfetch, ms, returned, labels_to_string(labels)});
                std::cout << name << " rep=" << repeat << " sel=" << sel << " q=" << qi << " ms=" << ms << " ret=" << returned << "\n";
            };

            for (int repeat = 0; repeat < repeats; repeat++) {
                if (includes_system(systems, "ACORN-faiss-gamma"))
                    run_acorn("ACORN-faiss-gamma", acorn, repeat);
                if (includes_system(systems, "ACORN-faiss-1"))
                    run_acorn("ACORN-faiss-1", acorn1, repeat);

                if (includes_system(systems, "FAISS-HNSW-sweeping")) {
                    labels.assign(k, -1);
                    dist.assign(k, 0);
                    auto t0 = Clock::now();
                    hnsw.search(1, q, k, dist.data(), labels.data(), &hnsw_params);
                    auto t1 = Clock::now();
                    int returned = 0;
                    for (auto x : labels) {
                        if (x >= 0) returned++;
                    }
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    results.push_back({"FAISS-HNSW-sweeping", repeat, sel, static_cast<int>(qi), qid, k, overfetch, ms, returned, labels_to_string(labels)});
                    std::cout << "FAISS-HNSW-sweeping rep=" << repeat << " sel=" << sel << " q=" << qi << " ms=" << ms << " ret=" << returned << "\n";
                }
            }
        }
    }

    write_detail(out, results);
    std::string summary = out;
    auto pos = summary.rfind(".csv");
    if (pos != std::string::npos) summary.replace(pos, 4, "_summary.csv");
    else summary += "_summary.csv";
    write_summary(summary, results);
    std::cerr << "wrote " << out << " and " << summary << "\n";
    return 0;
}
