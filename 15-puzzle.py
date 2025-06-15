import heapq
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
import os

os.makedirs("data", exist_ok=True)

# === Goal state and tile positions ===
GOAL_STATE = tuple([1, 2, 3, 4,
                    5, 6, 7, 8,
                    9,10,11,12,
                   13,14,15, 0])

GOAL_POSITIONS = {val: (val // 4, val % 4) for val in range(1, 16)}

def index_to_pos(index):
    return divmod(index, 4)

def swap(state, i, j):
    lst = list(state)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)

# === Heuristics ===

def manhattan_heuristic(state):
    total = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        r, c = index_to_pos(idx)
        gr, gc = GOAL_POSITIONS[tile]
        total += abs(r - gr) + abs(c - gc)
    return total

def euclidean_heuristic(state):
    total = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        r, c = index_to_pos(idx)
        gr, gc = GOAL_POSITIONS[tile]
        total += math.sqrt((r - gr) ** 2 + (c - gc) ** 2)
    return total

def misplaced_tiles_heuristic(state):
    return sum(1 for i, tile in enumerate(state) if tile != 0 and tile != GOAL_STATE[i])

def gaschnig_heuristic(state):
    state = list(state)
    goal = list(GOAL_STATE)
    cost = 0
    idx_map = {val: idx for idx, val in enumerate(state)}
    
    while state != goal:
        zero_idx = state.index(0)
        if goal[zero_idx] != 0:
            target_tile = goal[zero_idx]
            tile_idx = idx_map[target_tile]
            # Swap
            state[zero_idx], state[tile_idx] = state[tile_idx], state[zero_idx]
            idx_map[state[tile_idx]] = tile_idx
            idx_map[state[zero_idx]] = zero_idx
        else:
            # Find any misplaced tile
            for i in range(16):
                if state[i] != goal[i]:
                    state[zero_idx], state[i] = state[i], state[zero_idx]
                    idx_map[state[i]] = i
                    idx_map[state[zero_idx]] = zero_idx
                    break
        cost += 1
    return cost

def manhattan_linear_conflict(state):
    total = 0
    linear_conflict = 0

    # Build tile position grid
    grid = [state[i*4:(i+1)*4] for i in range(4)]

    for row in range(4):
        row_tiles = grid[row]
        goal_row_tiles = [((tile - 1) // 4) if tile != 0 else -1 for tile in row_tiles]
        for i in range(4):
            tile_i = row_tiles[i]
            if tile_i == 0 or goal_row_tiles[i] != row:
                continue
            for j in range(i + 1, 4):
                tile_j = row_tiles[j]
                if tile_j == 0 or goal_row_tiles[j] != row:
                    continue
                if tile_i > tile_j:
                    linear_conflict += 1

    for col in range(4):
        col_tiles = [grid[row][col] for row in range(4)]
        goal_col_tiles = [((tile - 1) % 4) if tile != 0 else -1 for tile in col_tiles]
        for i in range(4):
            tile_i = col_tiles[i]
            if tile_i == 0 or goal_col_tiles[i] != col:
                continue
            for j in range(i + 1, 4):
                tile_j = col_tiles[j]
                if tile_j == 0 or goal_col_tiles[j] != col:
                    continue
                if tile_i > tile_j:
                    linear_conflict += 1

    # Add Manhattan distances
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        r, c = divmod(idx, 4)
        gr, gc = GOAL_POSITIONS[tile]
        total += abs(r - gr) + abs(c - gc)

    return total + 2 * linear_conflict



# === A* Search ===

def get_neighbors(state):
    idx = state.index(0)
    r, c = index_to_pos(idx)
    moves = []

    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            n_idx = nr * 4 + nc
            neighbor = swap(state, idx, n_idx)
            moves.append(neighbor)

    return moves

def astar(start, heuristic_fn):
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start), 0, start))
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == GOAL_STATE:
            return cost, len(visited)

        visited.add(current)

        for neighbor in get_neighbors(current):
            tentative_g = cost + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                priority = tentative_g + heuristic_fn(neighbor)
                heapq.heappush(open_set, (priority, tentative_g, neighbor))


    return float('inf'), len(visited)

# === Generate Random States at Given Depth ===

def generate_states_bfs(max_depth=15, samples_per_depth=5):
    queue = deque([(GOAL_STATE, 0)])
    seen = {GOAL_STATE: 0}
    depth_bins = {i: [] for i in range(1, max_depth + 1)}

    while queue:
        state, depth = queue.popleft()

        if 1 <= depth <= max_depth and len(depth_bins[depth]) < samples_per_depth:
            depth_bins[depth].append(state)

        if all(len(bin) >= samples_per_depth for bin in depth_bins.values()):
            break

        if depth >= max_depth:
            continue

        for neighbor in get_neighbors(state):
            if neighbor not in seen:
                seen[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    for d in range(1, max_depth + 1):
        print(f"Depth {d}: {len(depth_bins[d])} samples")

    return depth_bins

# === Benchmarking Function ===

depth_bins = generate_states_bfs(max_depth=20, samples_per_depth=5)

def benchmark_heuristic(heuristic_fn, output_filename, depth_bins):
    records = []

    print(f"\nRunning benchmark for: {output_filename}")
    for depth in tqdm(range(1, 21)):
        if depth not in depth_bins or len(depth_bins[depth]) == 0:
            print(f"Skipping depth {depth}, not enough samples.")
            continue

        times, nodes = [], []

        for state in depth_bins[depth]:
            t0 = time.perf_counter_ns()
            cost, expanded = astar(state, heuristic_fn)
            t1 = time.perf_counter_ns()

            times.append(t1 - t0)
            nodes.append(expanded)

        records.append({
            "depth": depth,
            "avg_time_ns": np.mean(times),
            "avg_nodes_expanded": np.mean(nodes)
        })

    df = pd.DataFrame(records)
    df.to_csv(f"data/{output_filename}", index=False)

# === Run Benchmarks for All Heuristics ===

benchmark_heuristic(manhattan_heuristic, "15puzzle_manhattan.csv", depth_bins)
benchmark_heuristic(euclidean_heuristic, "15puzzle_euclidean.csv", depth_bins)
#benchmark_heuristic(misplaced_tiles_heuristic, "15puzzle_misplaced.csv", depth_bins)
#benchmark_heuristic(gaschnig_heuristic, "15puzzle_gaschnig.csv", depth_bins)
benchmark_heuristic(manhattan_linear_conflict, "15puzzle_manhattan_linear.csv", depth_bins)