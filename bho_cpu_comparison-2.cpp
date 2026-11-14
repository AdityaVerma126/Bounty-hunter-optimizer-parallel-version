// Sequential (single-thread) Bounty Hunter Optimizer -- paper-faithful implementation.
// Build : g++ -O3 -std=c++17 bho_cpu_comparison.cpp -o bho_cpu
// Run   : ./bho_cpu D N iterations repeats      (defaults 30 256 300 3)
#include <chrono>
#include <iomanip>
#include "bho_common.h"

int main(int argc, char** argv) {
    int D = argc > 1 ? std::atoi(argv[1]) : 30;
    int N = argc > 2 ? std::atoi(argv[2]) : 256;
    int max_iter = argc > 3 ? std::atoi(argv[3]) : 300;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 3;
    if (D < 2 || N < 4 || max_iter < 1 || repeats < 1) { std::cerr << "Usage: ./bho_cpu D N iterations repeats\n"; return 1; }
    Bench tests[10]; bho_benchmarks(tests);
    std::cout << "function,dimension,population,repeat,best_score,seconds\n";
    for (const auto& b : tests) for (int rep = 0; rep < repeats; rep++) {
        unsigned seed = 12345u + (unsigned)rep * 101u + (unsigned)b.id * 1009u;
        BhoCfg c = bho_make_cfg(D, N, max_iter, b, seed);
        auto t0 = std::chrono::steady_clock::now();
        double best = bho_run_host(c);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << b.name << "," << D << "," << N << "," << rep << "," << std::setprecision(12) << best << "," << sec << "\n";
    }
    return 0;
}
