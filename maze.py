import random, math, heapq, time, os
import numpy as np, pandas as pd
from collections import deque
from tqdm import tqdm

# === Config ===
GRID_SIZE = 30  # 30x30 grid
BLOCK_PROB = 0.3  # 30% of nodes are walls
MAX_DEPTH = 20
SAMPLES_PER_DEPTH = 1
MAX_ATTEMPTS = 1000
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
random.seed(42)

WALL, PATH = 1, 0

def generate_grid(size: int, block_prob: float) -> list[list[int]]:
    grid = [[PATH if random.random() > block_prob else WALL for _ in range(size)] for _ in range(size)]
    grid[0][0] = PATH  # Ensure start is walkable
    return grid

def bfs_distances(grid, start):
    rows, cols = len(grid), len(grid[0])
    distances = {}
    queue = deque([(start, 0)])
    visited = set([start])

    while queue:
        (r, c), d = queue.popleft()
        distances[(r, c)] = d
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == PATH and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return distances

def manhattan(p1, p2): return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
def euclidean(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def chebyshev(p1, p2): return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
def overestimated_euclidean(p1, p2): return 1.5 * euclidean(p1, p2)

heuristics = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev,
    "Overestimated Euclidean": overestimated_euclidean
}

def astar(grid, start, goal, heuristic_fn):
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start, goal), 0, start))
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return cost, len(visited)
        visited.add(current)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = current[0]+dr, current[1]+dc
            neighbor = (nr, nc)
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == PATH:
                tentative_g = cost + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + heuristic_fn(neighbor, goal)
                    heapq.heappush(open_set, (priority, tentative_g, neighbor))
    return float('inf'), len(visited)

def benchmark():
    start = (0, 0)
    goal_bank = {}
    grid_bank = {}

    print("Generating grid + goal sets per depth...")
    for depth in range(1, MAX_DEPTH + 1):
        for attempt in range(MAX_ATTEMPTS):
            grid = generate_grid(GRID_SIZE, BLOCK_PROB)
            distances = bfs_distances(grid, start)
            candidates = [pos for pos, d in distances.items() if d == depth]
            if len(candidates) >= SAMPLES_PER_DEPTH:
                goal_bank[depth] = random.sample(candidates, SAMPLES_PER_DEPTH)
                grid_bank[depth] = grid
                break
        else:
            print(f"⚠️ Could not find depth {depth} after {MAX_ATTEMPTS} attempts.")

    for name, fn in heuristics.items():
        records = []
        print(f"\nBenchmarking {name}")
        for depth in tqdm(range(1, MAX_DEPTH + 1)):
            if depth not in grid_bank:
                continue
            grid = grid_bank[depth]
            goals = goal_bank[depth]
            times, nodes, optimalities = [], [], []
            for goal in goals:
                optimal_cost, _ = astar(grid, start, goal, manhattan)
                t0 = time.perf_counter_ns()
                cost, expanded = astar(grid, start, goal, fn)
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
        df.to_csv(f"{DATA_DIR}/grid_{name.lower().replace(' ', '_')}.csv", index=False)

benchmark()
