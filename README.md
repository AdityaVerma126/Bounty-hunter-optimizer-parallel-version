
<div align="center">

# ⚡ Bounty Hunter Optimizer — Parallel Version

### Sequential CPU | OpenMP | CUDA GPU

**Accelerating Metaheuristic Optimization Through Parallel Computing**

<p>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=for-the-badge&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/OpenMP-Parallel%20CPU-green?style=for-the-badge" alt="OpenMP">
  <img src="https://img.shields.io/badge/CUDA-NVIDIA%20GPU-76B900?style=for-the-badge&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/Optimization-Metaheuristics-purple?style=for-the-badge" alt="Metaheuristics">
</p>

<p>
  <a href="https://github.com/AdityaVerma126/Bounty-hunter-optimizer-parallel-version">
    <strong>Explore Repository</strong>
  </a>
</p>

</div>

---

## 📌 Overview

This project explores the parallelization of a **Bounty Hunter Optimizer (BHO)-inspired metaheuristic** using three computational approaches:

- **Sequential C++** — baseline implementation.
- **OpenMP C++** — shared-memory CPU parallelization.
- **CUDA C++** — GPU-accelerated implementation using NVIDIA CUDA.

The objective is to compare execution time, optimization quality, and speedup across different workload sizes and benchmark functions.

The project demonstrates how parallel computing can accelerate population-based optimization algorithms and investigates the performance differences between multicore CPUs and GPUs.

> **Note:** This repository implements a BHO-inspired optimization variant for parallel performance experiments. It should not be considered an exact reproduction of every mechanism in the original Bounty Hunter Optimizer research paper.

## ✨ Key Features

- ⚙️ Sequential, OpenMP, and CUDA implementations.
- 🚀 GPU-based population updates and objective evaluation.
- 📊 Benchmarking across six standard optimization functions.
- 📈 Runtime and speedup comparisons.
- 🎯 Optimization-quality comparisons across implementations.
- 🔁 Repeated trials for more reliable performance analysis.
- 🧮 Configurable dimensions, population sizes, and iteration counts.
- ☁️ Google Colab workflow for GPU experiments.
- 📁 CSV-based experiment results and visualizations.

## 🧠 Optimization Algorithm

The optimizer maintains a population of candidate solutions. Each candidate represents a point in a continuous search space.

A typical optimization run follows these steps:

1. **Population initialization:** Generate candidate solutions within the search bounds.
2. **Fitness evaluation:** Evaluate each candidate using the objective function.
3. **Best-solution identification:** Find the best candidate in the population.
4. **Candidate generation:** Update candidates using best-guided movement, peer-relative differences, and random perturbations.
5. **Boundary handling:** Keep candidates within the permitted search space.
6. **Greedy selection:** Accept an updated candidate when it improves the objective value.
7. **Iteration:** Repeat the optimization process for the specified number of iterations.
8. **Result collection:** Record the best objective value and execution time.

The benchmark problems are minimization problems, so **lower objective values indicate better solutions**.

## ⚡ Parallelization Approaches

### 1. Sequential CPU

The sequential implementation executes the optimization steps serially.

- Provides the baseline execution time.
- Helps validate the optimization workflow.
- Serves as the reference for calculating speedup.

### 2. OpenMP CPU

The OpenMP implementation uses shared-memory parallelism to distribute independent population-level computations across CPU threads.

Parallelizable work includes population initialization, candidate generation, and candidate evaluation or selection where implemented.

The current best-candidate scan remains sequential in this implementation.

### 3. CUDA GPU

The CUDA implementation uses NVIDIA GPU threads to execute population-level operations in parallel.

The optimization loop keeps population data and intermediate arrays in GPU memory. CUDA kernels perform initialization, candidate generation and objective evaluation, and selection.

The current best-candidate scan is handled by a single CUDA thread, which may become a bottleneck for larger populations.

### Implementation comparison

| Feature | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Processor | CPU | Multicore CPU | NVIDIA GPU |
| Execution | Serial | Multithreaded | Massively parallel |
| Programming model | C++17 | C++17 + OpenMP | CUDA C++ |
| Population operations | Serial | Parallel loops | GPU kernels |
| Best-candidate scan | CPU | CPU | Single GPU thread |
| Main purpose | Baseline | CPU acceleration | GPU acceleration |

## 📊 Benchmark Functions

The project evaluates six standard continuous optimization functions.

| Function | Characteristics | Global minimum |
|---|---|---:|
| **Sphere** | Unimodal, convex | 0 |
| **Rastrigin** | Highly multimodal | 0 |
| **Rosenbrock** | Narrow curved valley | 0 |
| **Ackley** | Multimodal landscape | 0 |
| **Griewank** | Periodic multimodal landscape | 0 |
| **Schwefel** | Complex, deceptive landscape | Approximately 0 |

For Sphere, Rastrigin, Ackley, and Griewank, the global minimum is attained at the zero vector. Rosenbrock attains its global minimum at the all-ones vector. Schwefel's standard benchmark minimum is attained near \(x_i=420.9687\).

The theoretical global minimum provides a reference for evaluating solution quality; a finite stochastic run is not guaranteed to reach it.

### Search bounds

| Function | Lower bound | Upper bound |
|---|---:|---:|
| Sphere | -100 | 100 |
| Rastrigin | -5.12 | 5.12 |
| Rosenbrock | -30 | 30 |
| Ackley | -32.768 | 32.768 |
| Griewank | -600 | 600 |
| Schwefel | -500 | 500 |

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| C++17 | Sequential optimization implementation |
| OpenMP | CPU parallelization |
| CUDA C++ | GPU acceleration |
| Python | Benchmark orchestration and analysis |
| NumPy | Numerical operations |
| Pandas | Result aggregation and CSV processing |
| Matplotlib | Performance graphs |
| Google Colab | Cloud-based GPU execution |

## 🚀 Getting Started

### Recommended: Google Colab

The supplied benchmark notebook can be used to compile and run the implementations in a suitable Colab environment.

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Choose a GPU runtime, such as an available NVIDIA T4 GPU.
4. Run the setup and environment-check cells.
5. Confirm that the CUDA compiler and GPU are available.
6. Execute the compilation and benchmark cells.
7. Inspect the generated tables, graphs, and CSV results.

A GPU runtime must actually be allocated. Selecting a CPU-only runtime will not execute the CUDA implementation on an NVIDIA GPU.

### Local compilation

Use a compatible C++17 compiler for the CPU implementations and an installed NVIDIA CUDA Toolkit for the CUDA implementation.

**Sequential CPU**

```bash
g++ -O3 -std=c++17 bho_cpu_comparison.cpp -o bho_cpu
```

**OpenMP CPU**

```bash
g++ -O3 -std=c++17 -fopenmp bho_openmp.cpp -o bho_openmp
```

**CUDA GPU**

```bash
nvcc -O3 -std=c++17 bho_cuda.cu -o bho_cuda
```

The source filenames above correspond to the notebook's generated-source workflow. Make sure the source files exist in your working directory before compiling.

### Example execution

The benchmark programs accept dimension, population size, iteration count, and repeat count as positional arguments.

```bash
./bho_cpu 30 256 300 3
```

```bash
OMP_NUM_THREADS=4 ./bho_openmp 30 256 300 3
```

```bash
./bho_cuda 30 256 300 3
```

These commands illustrate a small benchmark configuration. Run them only after compiling the corresponding source files.

## ⚙️ Experiment Configuration

The notebook's benchmark configuration can be adjusted to investigate different workloads.

```python
D = 1000
POPULATION = 256
ITERATIONS = 300
REPEATS = 3
OMP_THREADS = 4
```

| Parameter | Description |
|---|---|
| `D` | Number of dimensions per candidate |
| `POPULATION` | Number of candidate solutions |
| `ITERATIONS` | Number of optimization iterations |
| `REPEATS` | Number of repeated trials |
| `OMP_THREADS` | Number of OpenMP threads |

Start with a smaller workload to verify correctness before increasing the dimension or population size.

For fair comparisons, keep the benchmark function, dimension, population size, iteration count, and repeat count consistent across implementations.

## 📈 Performance Evaluation

The project evaluates both computational performance and optimization quality.

### Metrics

| Metric | Meaning | Preferred result |
|---|---|---|
| Mean runtime | Average execution time | Lower |
| Runtime standard deviation | Variation in execution time | Lower |
| Mean best objective | Average best score across trials | Lower |
| Score standard deviation | Variation in optimization quality | Lower |
| Speedup | Sequential runtime divided by parallel runtime | Higher |

### Speedup calculation

\[
\text{Speedup} =
\frac{T_{\text{Sequential}}}
{T_{\text{Parallel}}}
\]

For example, if the sequential implementation takes 10 seconds and a parallel implementation takes 2 seconds:

\[
\text{Speedup}=\frac{10}{2}=5
\]

This represents a **5× speedup**.

A speedup greater than 1 indicates that the parallel implementation is faster for the measured workload. A speedup below 1 means it is slower.

### Expected visualizations

The benchmark notebook produces comparisons such as:

- Runtime comparison across sequential, OpenMP, and CUDA.
- CPU-to-OpenMP and CPU-to-CUDA speedup.
- Objective scores across benchmark functions.
- Mean and standard deviation of runtime and optimization results.

These graphs should be generated from actual benchmark results. No performance values are assumed or fabricated in this README.

## 📁 Benchmark Results

The notebook exports the following result files:

| File | Description |
|---|---|
| `bho_cpu_openmp_cuda_raw_trials.csv` | Per-trial benchmark measurements |
| `bho_cpu_openmp_cuda_summary.csv` | Mean and standard deviation summaries |
| `bho_cpu_openmp_cuda_comparison.csv` | Runtime, speedup, and objective-score comparisons |

The CSV files are generated by the benchmark workflow; they will be available after the relevant notebook cells run successfully.

## 📐 Scaling Experiments

To study how workload size affects parallel performance, consider experiments such as:

| Experiment | Dimension | Population | Iterations |
|---|---:|---:|---:|
| Small | 30 | 256 | 300 |
| Medium | 100 | 512 | 300 |
| Large | 300 | 1024 | 300 |
| High-dimensional | 1000 | 256 | 300 |

These are suggested configurations, not measured results.

Vary one parameter at a time where possible. Record the CPU model, GPU model, compiler versions, CUDA version, and OpenMP thread count.

## 🔍 Important Performance Considerations

- GPUs may be slower for small workloads because kernel-launch and synchronization overhead can dominate.
- Larger workloads may expose more parallelism, but speedup depends on hardware, memory access, and serial sections.
- Runtime and solution quality must be considered together.
- Stochastic algorithms should be evaluated over repeated trials.
- Use identical benchmark settings for meaningful comparisons.
- A serial best-candidate scan can limit GPU scalability.
- Report hardware and software configurations alongside results.

## 🔬 Future Improvements

Potential extensions include:

- Implement a parallel CUDA reduction for best-candidate identification.
- Study scaling across dimensions, population sizes, and iteration counts.
- Add convergence curves showing best-so-far objective value versus iteration.
- Increase the number of independent trials and add statistical comparisons.
- Investigate CUDA timing with events and distinguish kernel time from end-to-end runtime.
- Add more benchmark functions and higher-dimensional experiments.
- Compare with a verified implementation of the original Bounty Hunter Optimizer.
- Add automated checks for objective-function correctness and CPU/GPU result consistency.

## 📚 Reference

The project is inspired by research on the Bounty Hunter Optimizer and its optimization applications.

- [Bounty Hunter Optimizer — MATLAB Central reference page](https://www.mathworks.com/matlabcentral/fileexchange/183483-bounty-hunter-optimizer)

The parallel implementations in this repository are intended for studying performance and optimization behavior; consult the original publication for the exact algorithm definition.

## 🤝 Contributing

Contributions, suggestions, bug reports, and performance improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Document benchmark settings and hardware.
5. Submit a pull request with reproducible results.

## 📄 License

Check the repository for a `LICENSE` file before using or redistributing this project. If no license is present, reuse permissions should not be assumed.

---

<div align="center">

**Exploring parallel metaheuristic optimization with C++17, OpenMP, and CUDA.**

⭐ If you find this project useful, consider starring the repository.

[GitHub Repository](https://github.com/AdityaVerma126/Bounty-hunter-optimizer-parallel-version) · [Issues](https://github.com/AdityaVerma126/Bounty-hunter-optimizer-parallel-version/issues)

</div>
