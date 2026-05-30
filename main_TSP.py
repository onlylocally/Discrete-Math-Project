import sys
import os
import time
import random
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

try:
    import cupy as cp
except ImportError:
    cp = None

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

OPTIMAL_COST = 259045


def _array_module(value):
    if cp is not None and isinstance(value, cp.ndarray):
        return cp
    return np

def _to_python_int(value):
    if hasattr(value, 'item'):
        return int(value.item())
    return int(value)

def compute_dist_matrix(coords, use_gpu=True, block_size=1024):
    if use_gpu and cp is not None:
        coords_gpu = cp.asarray(coords, dtype=cp.float32)
        n = coords_gpu.shape[0]
        dist = cp.empty((n, n), dtype=cp.int32)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = coords_gpu[start:end, None, :] - coords_gpu[None, :, :]
            raw = cp.sqrt(cp.sum(block * block, axis=2))
            dist[start:end] = cp.floor(raw + 0.5).astype(cp.int32)
        return dist
    raw = cdist(coords, coords, 'euclidean')
    return np.floor(raw + 0.5).astype(np.int32)

def parse_tsp_coords(filepath):
    coords = []
    in_section = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith('NODE_COORD_SECTION'):
                in_section = True
                continue
            if line.upper().startswith('EOF'):
                break
            if in_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords.append((float(parts[1]), float(parts[2])))
                    except ValueError:
                        continue
    return np.array(coords)


class TSPSolver:

    def __init__(self, filepath):
        self.coords = parse_tsp_coords(filepath)
        self.n = len(self.coords)
        self.D = compute_dist_matrix(self.coords)

    # UTILS

    def calculate_total_cost(self, tour):
        if not tour:
            return 0
        xp = _array_module(self.D)
        base = tour if tour[0] == tour[-1] else tour + [tour[0]]
        idx = xp.asarray(base, dtype=xp.int64)
        total = xp.sum(self.D[idx[:-1], idx[1:]])
        return _to_python_int(total)

    # NEAREST NEIGHBOR (NN)

    def nearest_neighbor_tsp(self, start=0):
        n = len(self.D)
        xp = _array_module(self.D)
        D_float = self.D.astype(xp.float32, copy=True)
        visited = xp.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            dists = D_float[cur].copy()
            dists[visited] = xp.inf
            nxt = _to_python_int(xp.argmin(dists))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        tour.append(tour[0])
        return tour

    # FARTHEST INSERTION (FI)

    def nearest_pair_seed(self):
        n = len(self.D)
        if n == 0:
            return []
        xp = _array_module(self.D)
        D_float = self.D.astype(xp.float32, copy=True)
        diag = xp.arange(n)
        D_float[diag, diag] = xp.inf
        flat_idx = xp.argmin(D_float)
        i, j = xp.unravel_index(flat_idx, D_float.shape)
        return [int(i), int(j), int(i)]

    def best_insertion_position_vectorized(self, tour, city):
        xp = _array_module(self.D)
        base = tour[:-1] if tour and tour[0] == tour[-1] else tour[:]
        base_arr = xp.asarray(base, dtype=xp.int64)
        next_arr = xp.roll(base_arr, -1)
        deltas = self.D[base_arr, city] + self.D[city, next_arr] - self.D[base_arr, next_arr]
        best_idx = _to_python_int(xp.argmin(deltas))
        return (best_idx + 1, _to_python_int(deltas[best_idx]))

    def farthest_insertion(self):
        n = len(self.D)
        if n <= 2:
            return [0, 0] if n == 1 else [0, 1, 0]
        xp = _array_module(self.D)
        tour = self.nearest_pair_seed()
        in_tour = xp.zeros(n, dtype=bool)
        in_tour[tour[0]] = True
        in_tour[tour[1]] = True
        min_dist = xp.minimum(self.D[tour[0]].astype(xp.float32), self.D[tour[1]].astype(xp.float32))
        min_dist[in_tour] = -xp.inf
        remaining_count = n - 2
        while remaining_count > 0:
            best_city = _to_python_int(xp.argmax(min_dist))
            pos, _ = self.best_insertion_position_vectorized(tour, best_city)
            tour.insert(pos, best_city)
            in_tour[best_city] = True
            min_dist[best_city] = -xp.inf
            new_dists = self.D[best_city].astype(xp.float32)
            new_dists[in_tour] = -xp.inf
            min_dist = xp.minimum(min_dist, new_dists)
            remaining_count -= 1
        if tour[-1] != tour[0]:
            tour.append(tour[0])
        return tour

    # LOCAL SEARCH - 2-OPT

    def two_opt_vectorized(self, tour, strategy='best', verbose=0):
        xp = _array_module(self.D)
        n = len(tour)
        tour_arr = xp.array(tour, dtype=xp.int32)
        current_cost = self.calculate_total_cost(tour)
        iteration = 0
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = (-1, -1)
            for i in range(1, n - 2):
                node_i_prev = tour_arr[i - 1]
                node_i = tour_arr[i]
                j_arr = xp.arange(i + 1, n - 1)
                node_j = tour_arr[j_arr]
                node_j_next = tour_arr[j_arr + 1]
                deltas = self.D[node_i_prev, node_j] + self.D[node_i, node_j_next] - (self.D[node_i_prev, node_i] + self.D[node_j, node_j_next])
                min_idx = xp.argmin(deltas)
                min_delta = _to_python_int(deltas[min_idx])
                if min_delta < 0:
                    if strategy == 'first':
                        j_best = _to_python_int(j_arr[min_idx])
                        tour_arr[i:j_best + 1] = tour_arr[i:j_best + 1][::-1]
                        improved = True
                        current_cost += min_delta
                        if verbose == 1:
                            iteration += 1
                            print(f' [2-Opt First] Iter {iteration} | Delta: {min_delta} | New Cost: {current_cost}')
                        break
                    elif min_delta < best_delta:
                        best_delta = min_delta
                        best_i = i
                        best_j = _to_python_int(j_arr[min_idx])
            if strategy == 'best' and best_delta < 0:
                tour_arr[best_i:best_j + 1] = tour_arr[best_i:best_j + 1][::-1]
                improved = True
                current_cost += best_delta
                if verbose == 1:
                    iteration += 1
                    print(f' [2-Opt Best] Iter {iteration} | Delta: {best_delta} | New Cost: {current_cost}')
        return (tour_arr.tolist(), current_cost)

    # LOCAL SEARCH - 3-OPT 

    @staticmethod
    @njit(fastmath=True)
    def _three_opt_numba_core(tour_arr, D, strategy_is_first, verbose):
        n = len(tour_arr)
        improved = True
        current_cost = 0
        for idx in range(n - 1):
            current_cost += D[tour_arr[idx], tour_arr[idx + 1]]
        iteration = 0
        while improved:
            improved = False
            best_delta = 0
            best_move = (-1, -1, -1, -1)
            for i in range(1, n - 4):
                A = tour_arr[i - 1]
                B = tour_arr[i]
                for j in range(i + 2, n - 2):
                    C = tour_arr[j - 1]
                    D_node = tour_arr[j]
                    for k in range(j + 2, n):
                        E = tour_arr[k - 1]
                        F = tour_arr[k]
                        d0 = D[A, B] + D[C, D_node] + D[E, F]
                        d1 = D[A, C] + D[B, D_node] + D[E, F] - d0
                        d2 = D[A, B] + D[C, E] + D[D_node, F] - d0
                        d3 = D[A, C] + D[B, E] + D[D_node, F] - d0
                        d4 = D[A, D_node] + D[E, B] + D[C, F] - d0
                        d5 = D[A, D_node] + D[E, C] + D[B, F] - d0
                        d6 = D[A, E] + D[D_node, B] + D[C, F] - d0
                        d7 = D[A, E] + D[D_node, C] + D[B, F] - d0
                        if strategy_is_first:
                            if d1 < 0:
                                best_delta = d1
                                best_move = (i, j, k, 0)
                                break
                            if d2 < 0:
                                best_delta = d2
                                best_move = (i, j, k, 1)
                                break
                            if d3 < 0:
                                best_delta = d3
                                best_move = (i, j, k, 2)
                                break
                            if d4 < 0:
                                best_delta = d4
                                best_move = (i, j, k, 3)
                                break
                            if d5 < 0:
                                best_delta = d5
                                best_move = (i, j, k, 4)
                                break
                            if d6 < 0:
                                best_delta = d6
                                best_move = (i, j, k, 5)
                                break
                            if d7 < 0:
                                best_delta = d7
                                best_move = (i, j, k, 6)
                                break
                        else:
                            if d1 < best_delta:
                                best_delta = d1
                                best_move = (i, j, k, 0)
                            if d2 < best_delta:
                                best_delta = d2
                                best_move = (i, j, k, 1)
                            if d3 < best_delta:
                                best_delta = d3
                                best_move = (i, j, k, 2)
                            if d4 < best_delta:
                                best_delta = d4
                                best_move = (i, j, k, 3)
                            if d5 < best_delta:
                                best_delta = d5
                                best_move = (i, j, k, 4)
                            if d6 < best_delta:
                                best_delta = d6
                                best_move = (i, j, k, 5)
                            if d7 < best_delta:
                                best_delta = d7
                                best_move = (i, j, k, 6)
                    if strategy_is_first and best_delta < 0:
                        break
                if strategy_is_first and best_delta < 0:
                    break
            if best_delta < 0:
                i, j, k, case = best_move
                new_tour = np.empty_like(tour_arr)
                new_tour[:i] = tour_arr[:i]
                Y = tour_arr[i:j]
                Z = tour_arr[j:k]
                W = tour_arr[k:]
                if case == 0:
                    new_tour[i:j] = Y[::-1]
                    new_tour[j:k] = Z
                    new_tour[k:] = W
                elif case == 1:
                    new_tour[i:j] = Y
                    new_tour[j:k] = Z[::-1]
                    new_tour[k:] = W
                elif case == 2:
                    new_tour[i:j] = Y[::-1]
                    new_tour[j:k] = Z[::-1]
                    new_tour[k:] = W
                elif case == 3:
                    len_Z = len(Z)
                    new_tour[i:i + len_Z] = Z
                    new_tour[i + len_Z:k] = Y
                    new_tour[k:] = W
                elif case == 4:
                    len_Z = len(Z)
                    new_tour[i:i + len_Z] = Z
                    new_tour[i + len_Z:k] = Y[::-1]
                    new_tour[k:] = W
                elif case == 5:
                    len_Z = len(Z)
                    new_tour[i:i + len_Z] = Z[::-1]
                    new_tour[i + len_Z:k] = Y
                    new_tour[k:] = W
                elif case == 6:
                    len_Z = len(Z)
                    new_tour[i:i + len_Z] = Z[::-1]
                    new_tour[i + len_Z:k] = Y[::-1]
                    new_tour[k:] = W
                tour_arr = new_tour
                improved = True
                current_cost += best_delta
                iteration += 1
                if verbose == 1:
                    if strategy_is_first:
                        print(' [3-Opt Numba First] Iter', iteration, '| Case:', case + 1, '| Delta:', best_delta, '| Cost:', current_cost)
                    else:
                        print(' [3-Opt Numba Best] Iter', iteration, '| Case:', case + 1, '| Delta:', best_delta, '| Cost:', current_cost)
        return (tour_arr, current_cost)

    def three_opt_optimized(self, tour, strategy='best', verbose=0):
        if cp is not None and isinstance(self.D, cp.ndarray):
            D_cpu = cp.asnumpy(self.D)
        else:
            D_cpu = np.asarray(self.D, dtype=np.int32)
        tour_arr = np.asarray(tour, dtype=np.int32)
        strategy_is_first = strategy == 'first'
        optimized_tour_arr, final_cost = TSPSolver._three_opt_numba_core(tour_arr, D_cpu, strategy_is_first, verbose)
        return (optimized_tour_arr.tolist(), final_cost)

    # LOCAL SEARCH - OR-OPT

    @staticmethod
    @njit(fastmath=True)
    def _or_opt_core(tour_arr, D, max_chain=3):
        n = len(tour_arr)
        current_cost = 0
        for idx in range(n - 1):
            current_cost += D[tour_arr[idx], tour_arr[idx + 1]]
        improved = True
        iteration = 0
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            best_k = -1
            for chain_len in range(max_chain, 0, -1):
                for i in range(1, n - chain_len - 1):
                    prev_i = tour_arr[i - 1]
                    first_chain = tour_arr[i]
                    last_chain = tour_arr[i + chain_len - 1]
                    next_after_chain = tour_arr[i + chain_len]
                    remove_cost = D[prev_i, first_chain] + D[last_chain, next_after_chain] - D[prev_i, next_after_chain]
                    for j in range(0, n - 1):
                        if j >= i - 1 and j <= i + chain_len - 1:
                            continue
                        node_j = tour_arr[j]
                        node_j_next = tour_arr[j + 1]
                        insert_cost = D[node_j, first_chain] + D[last_chain, node_j_next] - D[node_j, node_j_next]
                        delta = insert_cost - remove_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                            best_k = chain_len
            if best_delta < 0:
                chain = tour_arr[best_i:best_i + best_k].copy()
                new_tour = np.empty(n, dtype=tour_arr.dtype)
                idx_out = 0
                for idx_in in range(n):
                    if idx_in < best_i or idx_in >= best_i + best_k:
                        new_tour[idx_out] = tour_arr[idx_in]
                        idx_out += 1
                if best_j >= best_i + best_k:
                    insert_after = best_j - best_k
                else:
                    insert_after = best_j
                final_tour = np.empty(n, dtype=tour_arr.dtype)
                for idx_in in range(insert_after + 1):
                    final_tour[idx_in] = new_tour[idx_in]
                for idx_in in range(best_k):
                    final_tour[insert_after + 1 + idx_in] = chain[idx_in]
                for idx_in in range(insert_after + 1, n - best_k):
                    final_tour[best_k + idx_in] = new_tour[idx_in]
                tour_arr = final_tour
                current_cost += best_delta
                improved = True
                iteration += 1
                if iteration % 50 == 0:
                    print(' [Or-Opt] Iter', iteration, '| Delta:', best_delta, '| Cost:', current_cost)
        return (tour_arr, current_cost)

    def or_opt_optimized(self, tour, max_chain=3):
        if cp is not None and hasattr(self.D, 'get'):
            D_cpu = self.D.get()
        else:
            D_cpu = np.asarray(self.D, dtype=np.int32)
        tour_arr = np.asarray(tour, dtype=np.int32)
        opt_arr, cost = TSPSolver._or_opt_core(tour_arr, D_cpu, max_chain)
        return (opt_arr.tolist(), int(cost))

    # DOUBLE BRIDGE

    def double_bridge(self, tour):
        if tour[0] == tour[-1]:
            t = tour[:-1]
        else:
            t = tour[:]
        n = len(t)
        pos = sorted(random.sample(range(1, n), 3))
        a, b, c = pos
        new_tour = t[:a] + t[b:c] + t[a:b] + t[c:]
        new_tour.append(new_tour[0])
        return new_tour

    # ILS CHAINED

    def ils_chained(self, time_limit=600, init_tour=None, verbose=True):
        start_time = time.time()
        if init_tour is None:
            init_tour = self.farthest_insertion()
        if verbose:
            print(' [ILS-Chain] Initial: 2-Opt Or-Opt 3-Opt...')
        t, c = self.two_opt_vectorized(init_tour, strategy='first', verbose=0)
        t, c = self.or_opt_optimized(t, max_chain=3)
        t, c = self.three_opt_optimized(t, strategy='first', verbose=0)
        best_tour = t[:]
        best_cost = c
        current_tour = t[:]
        if verbose:
            print(f' [ILS-Chain] Initial: {best_cost:,} | Gap: {(best_cost / OPTIMAL_COST - 1) * 100:.2f}%')
        iteration = 0
        no_improve = 0
        while time.time() - start_time < time_limit:
            iteration += 1
            perturbed = self.double_bridge(current_tour)
            t, c = self.two_opt_vectorized(perturbed, strategy='first', verbose=0)
            t, c = self.or_opt_optimized(t, max_chain=3)
            remaining = time_limit - (time.time() - start_time)
            if remaining > 30:
                t, c = self.three_opt_optimized(t, strategy='first', verbose=0)
            if c < best_cost:
                best_tour = t[:]
                best_cost = c
                current_tour = t
                no_improve = 0
                elapsed = time.time() - start_time
                if verbose:
                    print(f' [ILS-Chain] Iter {iteration} | NEW BEST: {best_cost:,} | Gap: {(best_cost / OPTIMAL_COST - 1) * 100:.2f}% | Time: {elapsed:.1f}s')
            else:
                no_improve += 1
                if no_improve > 3:
                    current_tour = t
                    no_improve = 0
        elapsed = time.time() - start_time
        if verbose:
            print(f' [ILS-Chain] DONE — {iteration} iterations in {elapsed:.1f}s')
            print(f' [ILS-Chain] Best: {best_cost:,} | Gap: {(best_cost / OPTIMAL_COST - 1) * 100:.2f}%')
        return (best_tour, best_cost)


# MAIN RUNNER

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsp_file = os.path.join(script_dir, 'pr1002.tsp')
    print('=' * 70)
    print(' ADVANCED TSP OPTIMIZATION (OOP) — pr1002 (Optimal: 259,045)')
    print('=' * 70)

    t0 = time.time()
    solver = TSPSolver(tsp_file)
    print(f'\nLoaded {solver.n} cities.')
    print(f'Distance matrix + init: {time.time() - t0:.2f}s')

    print('\n--- Initialize base tours ---')
    t0 = time.time()
    nn_tour = solver.nearest_neighbor_tsp(start=0)
    nn_cost = solver.calculate_total_cost(nn_tour)
    print(f'NN: {nn_cost:,} ({time.time() - t0:.2f}s)')

    t0 = time.time()
    fi_tour = solver.farthest_insertion()
    fi_cost = solver.calculate_total_cost(fi_tour)
    print(f'FI: {fi_cost:,} ({time.time() - t0:.2f}s)')

    results = []

    print('\n' + '=' * 70)
    print(' EXPERIMENT 1: Or-Opt')
    print('=' * 70)
    t0 = time.time()
    print('\n[1a] FI + Or-Opt(3)...')
    fi_oropt_tour, fi_oropt_cost = solver.or_opt_optimized(fi_tour, max_chain=3)
    fi_oropt_time = time.time() - t0
    print(f' => Cost: {fi_oropt_cost:,} | Gap: {(fi_oropt_cost / OPTIMAL_COST - 1) * 100:.2f}% | Time: {fi_oropt_time:.2f}s')
    results.append(('FI + Or-Opt(3)', fi_oropt_cost, fi_oropt_time))

    t0 = time.time()
    print('\n[1b] NN + 2-Opt First + Or-Opt(3)...')
    nn_2opt, nn_2opt_c = solver.two_opt_vectorized(nn_tour, strategy='first', verbose=0)
    nn_oropt_tour, nn_oropt_cost = solver.or_opt_optimized(nn_2opt, max_chain=3)
    nn_oropt_time = time.time() - t0
    print(f' => Cost: {nn_oropt_cost:,} | Gap: {(nn_oropt_cost / OPTIMAL_COST - 1) * 100:.2f}% | Time: {nn_oropt_time:.2f}s')
    results.append(('NN + 2-Opt + Or-Opt(3)', nn_oropt_cost, nn_oropt_time))

    print('\n' + '=' * 70)
    print(' EXPERIMENT 6: ILS Chained — 2-Opt Or-Opt 3-Opt (10 min)')
    print('=' * 70)
    t0 = time.time()
    ils_chain_tour, ils_chain_cost = solver.ils_chained(time_limit=600, init_tour=fi_tour)
    ils_chain_time = time.time() - t0
    results.append(('ILS-Chained (10min)', ils_chain_cost, ils_chain_time))

    print('\n' + '=' * 70)
    print(' EXPERIMENT 7: ILS Chained from NN (10 min)')
    print('=' * 70)
    t0 = time.time()
    ils_nn_tour, ils_nn_cost = solver.ils_chained(time_limit=600, init_tour=nn_tour)
    ils_nn_time = time.time() - t0
    results.append(('ILS-Chained from NN (10min)', ils_nn_cost, ils_nn_time))

    print('\n' + '=' * 70)
    print('  RESULTS SUMMARY')
    print('=' * 70)
    all_results = [
        ('Nearest Neighbor (baseline)', nn_cost, 0.05),
        ('Farthest Insertion (baseline)', fi_cost, 0.05),
        ('NN + 3-Opt Best (previous best)', 270823, 1210)
    ] + results
    all_results.sort(key=lambda x: x[1])
    print(f"\n{'Hang':<5} {'Cau hinh':<40} {'Cost':>10} {'Gap':>8} {'Time':>10}")
    print('-' * 78)
    for rank, (name, cost, t) in enumerate(all_results, 1):
        gap = (cost / OPTIMAL_COST - 1) * 100
        if t < 1:
            time_str = f'{t * 1000:.0f}ms'
        elif t < 60:
            time_str = f'{t:.1f}s'
        else:
            time_str = f'{t / 60:.1f}min'
        marker = ' ' if cost <= 260000 else ''
        print(f'{rank:<5} {name:<40} {cost:>10,} {gap:>7.2f}% {time_str:>10}{marker}')
    print(f'\n Optimal: {OPTIMAL_COST:,}')
    print(f' Best found: {all_results[0][1]:,} (Gap: {(all_results[0][1] / OPTIMAL_COST - 1) * 100:.2f}%)')
    print('=' * 70)
    print('\n' + '=' * 70)

# LKH

    print(' EXPERIMENT 8: LKH (elkai)')
    print('=' * 70)
    try:
        import elkai
        print('\nRunning LKH...')
        t0_lkh = time.time()
        lkh_tour = elkai.solve_int_matrix(solver.D.tolist())
        lkh_time = time.time() - t0_lkh
        lkh_cost = solver.calculate_total_cost(lkh_tour)
        print(f' => Cost: {lkh_cost:,} | Gap: {(lkh_cost / OPTIMAL_COST - 1) * 100:.2f}% | Time: {lkh_time:.2f}s')
    except ImportError:
        print('error LKH.')