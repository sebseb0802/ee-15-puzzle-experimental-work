import random, heapq, time, os
import numpy as np, pandas as pd
from collections import deque
from tqdm import tqdm
from copy import deepcopy

# === Config ===
MAX_DEPTH = 20
SAMPLES_PER_DEPTH = 5
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
random.seed(42)

# === Goal state ===
GOAL = ((1, 2, 3), (4, 5, 6), (7, 8, 0))  # 0 is the blank

# === Helpers ===
def to_tuple(board):
    return tuple(tuple(row) for row in board)

def find_blank(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j

# === Move generation ===
def get_neighbors(board):
    r, c = find_blank(board)
    moves = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_board = [list(row) for row in board]
            new_board[r][c], new_board[nr][nc] = new_board[nr][nc], new_board[r][c]
            moves.append(to_tuple(new_board))
    return moves

# === Generate solvable board at exact depth ===
def generate_board_at_depth(depth):
    visited = set()
    queue = deque([(GOAL, 0)])
    visited.add(GOAL)
    result = []
    while queue:
        board, d = queue.popleft()
        if d == depth:
            result.append(board)
            if len(result) >= SAMPLES_PER_DEPTH:
                return result
        if d > depth:
            break
        for neighbor in get_neighbors(board):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
    return result

# === Heuristics ===
def hamming(state):
    return sum(
        1 for i in range(3) for j in range(3)
        if state[i][j] != 0 and state[i][j] != GOAL[i][j]
    )

def manhattan(state):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_i, goal_j = (val - 1) // 3, (val - 1) % 3
                dist += abs(i - goal_i) + abs(j - goal_j)
    return dist

def overestimated_manhattan(state):
    return 5 * manhattan(state)

heuristics = {
    "Hamming": hamming,
    "Manhattan": manhattan,
    "Overestimated Manhattan": overestimated_manhattan
}

# === A* ===
def astar(start, heuristic_fn):
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start), 0, start))
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == GOAL:
            return cost, len(visited)
        visited.add(current)
        for neighbor in get_neighbors(current):
            tentative_g = cost + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_fn(neighbor)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    return float('inf'), len(visited)

# === Benchmarking ===
def benchmark():
    depth_boards = {}
    print("Generating puzzles at each depth...")
    for depth in tqdm(range(1, MAX_DEPTH + 1)):
        boards = generate_board_at_depth(depth)
        if len(boards) == SAMPLES_PER_DEPTH:
            depth_boards[depth] = boards
        else:
            print(f"⚠️ Depth {depth}: only found {len(boards)} boards")

    for name, fn in heuristics.items():
        records = []
        print(f"\nBenchmarking {name}")
        for depth in tqdm(range(1, MAX_DEPTH + 1)):
            if depth not in depth_boards:
                continue
            puzzles = depth_boards[depth]
            times, nodes, optimalities = [], [], []
            for puzzle in puzzles:
                optimal_cost, _ = astar(puzzle, manhattan)
                t0 = time.perf_counter_ns()
                cost, expanded = astar(puzzle, fn)
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
        df.to_csv(f"{DATA_DIR}/eight_puzzle_{name.lower().replace(' ', '_')}.csv", index=False)

benchmark()
