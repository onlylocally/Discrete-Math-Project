import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_TSP import TSPSolver, OPTIMAL_COST, _to_python_int, _array_module

script_dir = os.path.dirname(os.path.abspath(__file__))
tsp_file = os.path.join(script_dir, '..', 'Dataset', 'pr1002.tsp')

ROUTE_COLOR = '#2980b9'
CITY_COLOR = '#e74c3c'
START_COLOR = '#27ae60'
ORIG_COLOR = 'gray'


def plot_single_route(coords, tour, title, cost, filename):
    if hasattr(tour, 'tolist'):
        tour = tour.tolist()
    fig, ax = plt.subplots(figsize=(12, 8))
    x = [coords[i][0] for i in tour]
    y = [coords[i][1] for i in tour]
    gap = (cost / OPTIMAL_COST - 1) * 100
    ax.plot(x, y, color=ROUTE_COLOR, linewidth=1.0, alpha=0.8, label=f'Route (Cost: {cost:,})')
    ax.scatter([coords[i][0] for i in tour[:-1]], [coords[i][1] for i in tour[:-1]],
               color=CITY_COLOR, s=8, zorder=5, label='Cities')
    ax.scatter(coords[tour[0]][0], coords[tour[0]][1],
               color=START_COLOR, s=80, marker='*', zorder=6, label='Start Node')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'{title}\nCost: {cost:,}  |  Gap: +{gap:.2f}%', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}.png')


def plot_comparison(coords, base_tour, opt_tour, base_name, opt_name, base_cost, opt_cost, filename):
    if hasattr(base_tour, 'tolist'):
        base_tour = base_tour.tolist()
    if hasattr(opt_tour, 'tolist'):
        opt_tour = opt_tour.tolist()
    fig, ax = plt.subplots(figsize=(12, 8))
    xb = [coords[i][0] for i in base_tour]
    yb = [coords[i][1] for i in base_tour]
    ax.plot(xb, yb, color=ORIG_COLOR, linestyle='--', linewidth=1.0, alpha=0.4,
            label=f'Original: {base_name} (Cost: {base_cost:,})')
    xo = [coords[i][0] for i in opt_tour]
    yo = [coords[i][1] for i in opt_tour]
    ax.plot(xo, yo, color=ROUTE_COLOR, linewidth=1.5, alpha=0.9,
            label=f'Optimized: {opt_name} (Cost: {opt_cost:,})')
    ax.scatter([coords[i][0] for i in opt_tour[:-1]], [coords[i][1] for i in opt_tour[:-1]],
               color=CITY_COLOR, s=8, zorder=5, label='Cities')
    ax.scatter(coords[opt_tour[0]][0], coords[opt_tour[0]][1],
               color=START_COLOR, s=80, marker='*', zorder=6, label='Start Node')
    improvement = base_cost - opt_cost
    pct = improvement / base_cost * 100
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Comparison: {base_name} vs {opt_name}\nImprovement: {improvement:,} cost units ({pct:.2f}%)',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}.png')


def two_opt_with_history(solver, tour, strategy='best'):
    xp = _array_module(solver.D)
    n = len(tour)
    tour_arr = xp.array(tour, dtype=xp.int32)
    current_cost = solver.calculate_total_cost(tour)
    history = [current_cost]
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
            deltas = solver.D[node_i_prev, node_j] + solver.D[node_i, node_j_next] - (solver.D[node_i_prev, node_i] + solver.D[node_j, node_j_next])
            min_idx = xp.argmin(deltas)
            min_delta = _to_python_int(deltas[min_idx])
            if min_delta < 0:
                if strategy == 'first':
                    j_best = _to_python_int(j_arr[min_idx])
                    tour_arr[i:j_best + 1] = tour_arr[i:j_best + 1][::-1]
                    improved = True
                    current_cost += min_delta
                    history.append(current_cost)
                    break
                elif min_delta < best_delta:
                    best_delta = min_delta
                    best_i = i
                    best_j = _to_python_int(j_arr[min_idx])
        if strategy == 'best' and best_delta < 0:
            tour_arr[best_i:best_j + 1] = tour_arr[best_i:best_j + 1][::-1]
            improved = True
            current_cost += best_delta
            history.append(current_cost)
    return tour_arr.tolist(), current_cost, history


from numba import njit
@njit(fastmath=True)
def _three_opt_history_core(tour_arr, D, strategy_is_first):
    n = len(tour_arr)
    current_cost = 0
    for idx in range(n - 1):
        current_cost += D[tour_arr[idx], tour_arr[idx + 1]]
    
    history = np.zeros(10000, dtype=np.int32)
    history[0] = current_cost
    history_idx = 1
    
    improved = True
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
                        if d1 < 0: best_delta = d1; best_move = (i, j, k, 0); break
                        if d2 < 0: best_delta = d2; best_move = (i, j, k, 1); break
                        if d3 < 0: best_delta = d3; best_move = (i, j, k, 2); break
                        if d4 < 0: best_delta = d4; best_move = (i, j, k, 3); break
                        if d5 < 0: best_delta = d5; best_move = (i, j, k, 4); break
                        if d6 < 0: best_delta = d6; best_move = (i, j, k, 5); break
                        if d7 < 0: best_delta = d7; best_move = (i, j, k, 6); break
                    else:
                        if d1 < best_delta: best_delta = d1; best_move = (i, j, k, 0)
                        if d2 < best_delta: best_delta = d2; best_move = (i, j, k, 1)
                        if d3 < best_delta: best_delta = d3; best_move = (i, j, k, 2)
                        if d4 < best_delta: best_delta = d4; best_move = (i, j, k, 3)
                        if d5 < best_delta: best_delta = d5; best_move = (i, j, k, 4)
                        if d6 < best_delta: best_delta = d6; best_move = (i, j, k, 5)
                        if d7 < best_delta: best_delta = d7; best_move = (i, j, k, 6)
                if strategy_is_first and best_delta < 0: break
            if strategy_is_first and best_delta < 0: break
        
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
                lz = len(Z)
                new_tour[i:i + lz] = Z
                new_tour[i + lz:k] = Y
                new_tour[k:] = W
            elif case == 4:
                lz = len(Z)
                new_tour[i:i + lz] = Z
                new_tour[i + lz:k] = Y[::-1]
                new_tour[k:] = W
            elif case == 5:
                lz = len(Z)
                new_tour[i:i + lz] = Z[::-1]
                new_tour[i + lz:k] = Y
                new_tour[k:] = W
            elif case == 6:
                lz = len(Z)
                new_tour[i:i + lz] = Z[::-1]
                new_tour[i + lz:k] = Y[::-1]
                new_tour[k:] = W
            tour_arr = new_tour
            improved = True
            current_cost += best_delta
            history[history_idx] = current_cost
            history_idx += 1
            if history_idx >= len(history):
                break
                
    return tour_arr, current_cost, history[:history_idx]

def three_opt_with_history(solver, tour, strategy='best'):
    if hasattr(solver.D, 'get'):
        D = solver.D.get()
    else:
        D = np.asarray(solver.D, dtype=np.int32)
    tour_arr = np.asarray(tour, dtype=np.int32)
    strategy_is_first = strategy == 'first'
    
    optimized_tour, final_cost, hist = _three_opt_history_core(tour_arr, D, strategy_is_first)
    return optimized_tour.tolist(), int(final_cost), hist.tolist()


def plot_convergence(nn_histories, fi_histories, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    nn_colors = ['#00008B', '#4169E1', '#1E90FF', '#87CEEB']
    fi_colors = ['#8B0000', '#CD5C5C', '#FF6B6B', '#FFB6C1']
    nn_labels = ['NN + 2-Opt (First Imp)', 'NN + 2-Opt (Best Imp)',
                 'NN + 3-Opt (First Imp)', 'NN + 3-Opt (Best Imp)']
    fi_labels = ['FI + 2-Opt (First Imp)', 'FI + 2-Opt (Best Imp)',
                 'FI + 3-Opt (First Imp)', 'FI + 3-Opt (Best Imp)']
    for i, (hist, label) in enumerate(zip(nn_histories, nn_labels)):
        gain = hist[0] - hist[-1]
        ax.plot(range(len(hist)), hist, color=nn_colors[i], linewidth=1.5,
                label=f'{label} (Gain: {gain:,})')
    for i, (hist, label) in enumerate(zip(fi_histories, fi_labels)):
        gain = hist[0] - hist[-1]
        ax.plot(range(len(hist)), hist, color=fi_colors[i], linewidth=1.5,
                label=f'{label} (Gain: {gain:,})')
    ax.set_xlabel('Improvement Steps')
    ax.set_ylabel('Total Cost')
    ax.set_title('Convergence Proof: Nearest Neighbor vs Farthest Insertion', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}.png')


if __name__ == '__main__':
    print('=' * 60)
    print(' CHART GENERATOR')
    print('=' * 60)

    print('\nInitializing solver...')
    solver = TSPSolver(tsp_file)
    coords = solver.coords
    print(f'Loaded {solver.n} cities.')

    print('\n--- Building base tours ---')
    nn_tour = solver.nearest_neighbor_tsp(start=0)
    nn_cost = solver.calculate_total_cost(nn_tour)
    print(f'NN: {nn_cost:,}')

    fi_tour = solver.farthest_insertion()
    fi_cost = solver.calculate_total_cost(fi_tour)
    print(f'FI: {fi_cost:,}')

    # Chart 1: NN
    print('\n[1/5] Nearest Neighbor standalone...')
    plot_single_route(coords, nn_tour, 'Nearest Neighbor', nn_cost, 'chart_nn')

    # Chart 2: FI
    print('[2/5] Farthest Insertion standalone...')
    plot_single_route(coords, fi_tour, 'Farthest Insertion', fi_cost, 'chart_fi')

    # Chart 3: NN vs NN+3-Opt (Best Imp) comparison
    print('[3/5] NN vs NN + 3-Opt (Best Imp) comparison...')
    nn_3opt_tour, nn_3opt_cost = solver.three_opt_optimized(nn_tour, strategy='best', verbose=0)
    print(f'  NN + 3-Opt Best: {nn_3opt_cost:,}')
    plot_comparison(coords, nn_tour, nn_3opt_tour,
                    'Nearest Neighbor', 'NN + 3-Opt (Best Imp)',
                    nn_cost, nn_3opt_cost, 'chart_nn_vs_nn3opt')

    # Chart 4: FI vs FI+3-Opt (Best Imp) comparison
    print('[4/5] FI vs FI + 3-Opt (Best Imp) comparison...')
    fi_3opt_tour, fi_3opt_cost = solver.three_opt_optimized(fi_tour, strategy='best', verbose=0)
    print(f'  FI + 3-Opt Best: {fi_3opt_cost:,}')
    plot_comparison(coords, fi_tour, fi_3opt_tour,
                    'Farthest Insertion', 'FI + 3-Opt (Best Imp)',
                    fi_cost, fi_3opt_cost, 'chart_fi_vs_fi3opt')

    # Chart 5: Convergence proof
    print('[5/5] Convergence chart (8 curves)...')
    print('  Running NN + 2-Opt First...')
    _, _, nn_2opt_first_h = two_opt_with_history(solver, nn_tour, strategy='first')
    print('  Running NN + 2-Opt Best...')
    _, _, nn_2opt_best_h = two_opt_with_history(solver, nn_tour, strategy='best')
    print('  Running NN + 3-Opt First...')
    _, _, nn_3opt_first_h = three_opt_with_history(solver, nn_tour, strategy='first')
    print('  Running NN + 3-Opt Best...')
    _, _, nn_3opt_best_h = three_opt_with_history(solver, nn_tour, strategy='best')
    print('  Running FI + 2-Opt First...')
    _, _, fi_2opt_first_h = two_opt_with_history(solver, fi_tour, strategy='first')
    print('  Running FI + 2-Opt Best...')
    _, _, fi_2opt_best_h = two_opt_with_history(solver, fi_tour, strategy='best')
    print('  Running FI + 3-Opt First...')
    _, _, fi_3opt_first_h = three_opt_with_history(solver, fi_tour, strategy='first')
    print('  Running FI + 3-Opt Best...')
    _, _, fi_3opt_best_h = three_opt_with_history(solver, fi_tour, strategy='best')

    nn_histories = [nn_2opt_first_h, nn_2opt_best_h, nn_3opt_first_h, nn_3opt_best_h]
    fi_histories = [fi_2opt_first_h, fi_2opt_best_h, fi_3opt_first_h, fi_3opt_best_h]
    plot_convergence(nn_histories, fi_histories, 'chart_convergence')
