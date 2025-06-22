import heapq
import time
import math
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# === Setup ===
GRID_SIZE = 40
MAX_DEPTH = 30
SAMPLES_PER_DEPTH = 5
START = (0, 0)

os.makedirs("data", exist_ok=True)

# === Heuristic Functions ===
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

def overestimated_euclidean(p1, p2):
    return 1.5 * euclidean(p1, p2)

# === A* Algorithm with Diagonal Movement ===
def astar(start, goal, heuristic_fn):
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start, goal), 0, start))
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return cost, len(visited)

        if current in visited:
            continue
        visited.add(current)

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                step_cost = math.sqrt(2) if abs(dr) == 1 and abs(dc) == 1 else 1
                tentative_g = cost + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + heuristic_fn(neighbor, goal)
                    heapq.heappush(open_set, (priority, tentative_g, neighbor))

    return float('inf'), len(visited)

# === Generate Goal States ===
def generate_goals_at_depths(max_depth, samples_per_depth):
    goals = {d: [] for d in range(1, max_depth + 1)}
    for d in range(1, max_depth + 1):
        for offset in range(samples_per_depth):
            r = min(GRID_SIZE - 1, d + offset)
            c = min(GRID_SIZE - 1, d + offset)
            goals[d].append((r, c))
    return goals

# === Benchmark Heuristic ===
def benchmark_heuristic(heuristic_fn, heuristic_name, goal_depths):
    records = []
    print(f"\nBenchmarking: {heuristic_name}")

    for depth in tqdm(range(1, MAX_DEPTH + 1)):
        times, nodes, optimalities = [], [], []
        for goal in goal_depths[depth]:
            optimal_cost, _ = astar(START, goal, euclidean)

            t0 = time.perf_counter_ns()
            cost, expanded = astar(START, goal, heuristic_fn)
            t1 = time.perf_counter_ns()

            times.append(t1 - t0)
            nodes.append(expanded)
            optimalities.append(cost / optimal_cost if optimal_cost > 0 else 1.0)

        records.append({
            "depth": depth,
            "avg_time_ns": np.mean(times),
            "avg_nodes_expanded": np.mean(nodes),
            "avg_optimality_ratio": np.mean(optimalities)
        })

    df = pd.DataFrame(records)
    filename = f"data/grid_{heuristic_name.lower().replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    return filename

# === Run All Benchmarks ===
goal_depths = generate_goals_at_depths(MAX_DEPTH, SAMPLES_PER_DEPTH)

benchmark_heuristic(manhattan, "Manhattan", goal_depths)
benchmark_heuristic(euclidean, "Euclidean", goal_depths)
benchmark_heuristic(chebyshev, "Chebyshev", goal_depths)
benchmark_heuristic(overestimated_euclidean, "Overestimated Euclidean", goal_depths)
