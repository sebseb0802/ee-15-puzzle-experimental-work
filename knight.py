import random, math, heapq, time, os
import numpy as np, pandas as pd
from collections import deque
from tqdm import tqdm

# === Config ===
BOARD_SIZE = 10
MAX_DEPTH = 20
SAMPLES_PER_DEPTH = 1
MAX_ATTEMPTS = 1000
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
random.seed(42)

# === Knight move set ===
MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
         (1, -2), (1, 2), (2, -1), (2, 1)]

def is_valid(pos):
    r, c = pos
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

# === BFS for ground truth depth ===
def bfs_distances(start):
    distances = {}
    queue = deque([(start, 0)])
    visited = set([start])
    
    while queue:
        (r, c), d = queue.popleft()
        distances[(r, c)] = d
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if is_valid((nr, nc)) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return distances

# === Heuristics ===
def manhattan(p1, p2): return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
def euclidean(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def chebyshev(p1, p2): return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
def overestimated_euclidean(p1, p2): return 3 * euclidean(p1, p2)

heuristics = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev,
    "Overestimated Euclidean": overestimated_euclidean
}

# === A* Search ===
def astar(start, goal, heuristic_fn):
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start, goal), 0, start))
    g_score = {start: 0}
    visited = set()
    
    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return cost, len(visited)
        visited.add(current)
        for dr, dc in MOVES:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if is_valid(neighbor):
                tentative_g = cost + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + heuristic_fn(neighbor, goal)
                    heapq.heappush(open_set, (priority, tentative_g, neighbor))
    return float('inf'), len(visited)

# === Benchmarking ===
def benchmark():
    start = (0, 0)
    goal_bank = {}
    
    print("Computing valid goals per depth...")
    distances = bfs_distances(start)
    for depth in range(1, MAX_DEPTH + 1):
        candidates = [pos for pos, d in distances.items() if d == depth]
        if len(candidates) >= SAMPLES_PER_DEPTH:
            goal_bank[depth] = random.sample(candidates, SAMPLES_PER_DEPTH)
        else:
            print(f"⚠️ Could not find enough nodes at depth {depth}. Skipping.")

    for name, fn in heuristics.items():
        records = []
        print(f"\nBenchmarking {name}")
        for depth in tqdm(range(1, MAX_DEPTH + 1)):
            if depth not in goal_bank:
                continue
            goals = goal_bank[depth]
            times, nodes, optimalities = [], [], []
            for goal in goals:
                optimal_cost, _ = astar(start, goal, manhattan)
                t0 = time.perf_counter_ns()
                cost, expanded = astar(start, goal, fn)
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
        df.to_csv(f"{DATA_DIR}/knight_{name.lower().replace(' ', '_')}.csv", index=False)

benchmark()
