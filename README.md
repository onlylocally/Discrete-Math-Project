# TSP — PR1002 Dataset

> **Benchmark:** PR1002 (1,002 cities) | **Optimal:** 259,045 | **Type:** EUC_2D

## Project Structure

```
RR/
├── TSP/
│   ├── main_TSP.py          # Core solver — TSPSolver class + all algorithms
│   └── generate_charts.py   # Generates 5 route and convergence charts as PNG
└── Dataset/
    └── pr1002.tsp           # TSPLIB Dataset (1,002 cities)
```

## Implementation Details

### Performance optimizations

| Technique | Applied to | Effect |
|---|---|---|
| NumPy vectorization | 2-Opt inner loop, FI insertion | Removes Python-level loop over $j$ |
| CuPy GPU backend | Distance matrix computation, 2-Opt | 10–50× speedup on CUDA hardware |
| Numba `@njit(fastmath=True)` | 3-Opt core, Or-Opt core | JIT-compiled to near-C speed |
| Incremental `min_dist` update | Farthest Insertion | $O(n)$ per step instead of $O(n^2)$ |
| Integer distance matrix | All local search | Avoids floating-point comparison issues |

## Algorithms

### Construction (Greedy)

| Algorithm | Idea | Complexity |
|---|---|---|
| Nearest Neighbor (NN) | Always go to the closest unvisited city | O(n²) |
| Farthest Insertion (FI) | Insert the city farthest from the current tour at the cheapest position | O(n²) |

### Local Search

| Algorithm | Idea | Complexity |
|---|---|---|
| 2-Opt | Remove 2 edges, reverse the segment between them | O(n²) per pass |
| 3-Opt | Remove 3 edges, try 7 reconnection patterns | O(n³) per pass |
| Or-Opt | Relocate a chain of 1–3 cities to a better position | O(n²) per pass |

Each local search supports two acceptance strategies:
- **First Improvement** — accept the first move that improves the tour, then restart
- **Best Improvement** — scan the full neighborhood, accept the single best move

### Metaheuristic

| Algorithm | Idea |
|---|---|
| ILS Chained | Repeatedly perturb (Double Bridge) then re-optimize with `2-Opt → Or-Opt → 3-Opt` |
| LKH (elkai) | State-of-the-art exact/near-optimal solver, used as reference upper bound |

---

### GPU/CPU transparency

```python
# Automatically uses CuPy if available, falls back to NumPy
xp = _array_module(self.D)   # returns cp or np
```

The `_array_module()` helper allows all array operations to run identically on CPU or GPU.

---

## Charts and Visualizations

Running `python generate_charts.py` produces 5 PNG files:

| Output file | Content |
|---|---|
| `chart_nn.png` | Route map — Nearest Neighbor tour |
| `chart_fi.png` | Route map — Farthest Insertion tour |
| `chart_nn_vs_nn3opt.png` | Overlay comparison: NN (gray dashed) vs NN + 3-Opt Best (blue) |
| `chart_fi_vs_fi3opt.png` | Overlay comparison: FI (gray dashed) vs FI + 3-Opt Best (blue) |
| `chart_convergence.png` | Convergence proof: 8 curves (NN/FI × 2-Opt/3-Opt × First/Best), cost vs. improvement steps |

---

## Installation and Usage

### Requirements

```
Python >= 3.9
numpy
scipy
matplotlib
numba
```

Optional (for GPU acceleration):
```
cupy-cuda12x   # or cupy-cuda11x depending on your CUDA version
```

Optional (for LKH reference):
```
elkai
```

> **⚠️ Lưu ý:** `elkai` hiện chỉ hỗ trợ **Python ≤ 3.12**. Trên Python 3.13+ package sẽ không build được do xung đột với `scikit-build-core`. Nếu bạn dùng Python 3.13+, hãy tạo một virtual environment với Python 3.12 để chạy LKH, hoặc bỏ qua bước này — chương trình sẽ tự động skip LKH nếu `elkai` không được cài.

### Install

```bash
pip install numpy scipy matplotlib numba
# Optional GPU:
pip install cupy-cuda12x
# Optional LKH (yêu cầu Python ≤ 3.12):
pip install elkai
```

### Run all experiments

```bash
cd TSP
python main_TSP.py
```

Output: ranked results table in terminal with cost, gap to optimal, and runtime for each configuration.

### Generate charts

```bash
cd TSP
python generate_charts.py
```

Output: 5 PNG files saved to `TSP/` directory. Expect ~5–30 minutes depending on hardware
(3-Opt runs are the bottleneck).

---

## Discussion

### Construction quality

Farthest Insertion consistently produces better initial tours than Nearest Neighbor on PR1002.
By inserting the most remote cities first, it avoids the long "closing edge" problem that
plagues Nearest Neighbor when distant cities are left to the end. However, NN is significantly
faster and serves as a reasonable starting point for metaheuristics with many restarts.

### Local search trade-offs

- **2-Opt vs 3-Opt:** 3-Opt produces substantially shorter tours but at $O(n^3)$ cost per
  iteration. On PR1002 with 1,002 cities, one 3-Opt pass (Best Improvement) can take tens of
  minutes. The Numba JIT compilation is essential to make this feasible.

- **First vs Best Improvement:** For 2-Opt, First Improvement often converges faster in wall-clock
  time due to fewer evaluations per improvement. For 3-Opt, the difference is less clear since
  each iteration is already expensive.

- **Or-Opt:** Provides a good intermediate step — stronger than 2-Opt alone in terms of move
  diversity, but much cheaper than 3-Opt. Chaining `2-Opt → Or-Opt → 3-Opt` in ILS gives a
  strong local minimum before perturbation.

### ILS and escaping local optima

The Double Bridge perturbation is specifically designed so that its effects cannot be undone by
2-Opt alone, forcing genuine exploration of new regions of the search space. This property makes
ILS with Double Bridge significantly more effective than pure iterated 2-Opt restarts.

### Complexity summary

| Method | Time complexity | Space complexity |
|---|---|---|
| Nearest Neighbor | $O(n^2)$ | $O(n)$ |
| Farthest Insertion | $O(n^2)$ | $O(n)$ |
| 2-Opt (one pass) | $O(n^2)$ | $O(n)$ |
| 3-Opt (one pass) | $O(n^3)$ | $O(n)$ |
| Or-Opt (one pass) | $O(n^2)$ | $O(n)$ |
| Distance matrix | $O(n^2)$ | $O(n^2)$ |

### Real-world applicability

TSP has direct applications in **Logistics and Supply Chain Management**, including:
- Last-mile delivery route optimization
- Warehouse picking path planning
- PCB drilling path minimization
- DNA fragment assembly

In practice, near-optimal solutions (gap < 5%) found quickly are often more valuable than
mathematically optimal solutions found slowly. The ILS-Chained approach in this project
demonstrates that strong, practical solutions can be obtained within a fixed time budget —
a common real-world constraint.
