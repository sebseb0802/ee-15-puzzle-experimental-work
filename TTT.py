import random
import time
import pandas as pd
from typing import List, Optional

# ==== Game logic ====
Player = str
Board = List[Optional[Player]]

def initial_board() -> Board:
    return [None] * 9

def is_winner(b: Board, p: Player) -> bool:
    wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    return any(all(b[i] == p for i in line) for line in wins)

def is_draw(b: Board) -> bool:
    return all(cell is not None for cell in b) and not is_winner(b, 'X') and not is_winner(b, 'O')

def legal_moves(b: Board) -> List[int]:
    return [i for i, cell in enumerate(b) if cell is None]

def apply_move(b: Board, move: int, player: Player) -> Board:
    new_board = b.copy()
    new_board[move] = player
    return new_board

def evaluate(b: Board, p: Player) -> int:
    opponent = 'O' if p == 'X' else 'X'
    if is_winner(b, p):
        return 1
    elif is_winner(b, opponent):
        return -1
    else:
        return 0

# ==== Heuristic Function ====
def simple_heuristic(board: Board, player: Player) -> int:
    opponent = 'O' if player == 'X' else 'X'
    score = 0
    lines = [[0,1,2],[3,4,5],[6,7,8],
             [0,3,6],[1,4,7],[2,5,8],
             [0,4,8],[2,4,6]]
    for line in lines:
        values = [board[i] for i in line]
        if values.count(player) == 2 and values.count(None) == 1:
            score += 10
        if values.count(opponent) == 2 and values.count(None) == 1:
            score -= 8
        if values.count(player) == 1 and values.count(None) == 2:
            score += 1
    return score

# ==== Full-depth Minimax ====
def minimax(b: Board, d: int, maximizing: bool, p: Player, nodes: List[int]) -> int:
    nodes[0] += 1
    if d == 0 or is_draw(b) or is_winner(b, 'X') or is_winner(b, 'O'):
        return evaluate(b, p)
    opponent = 'O' if p == 'X' else 'X'
    if maximizing:
        best = -float('inf')
        for move in legal_moves(b):
            val = minimax(apply_move(b, move, p), d - 1, False, p, nodes)
            best = max(best, val)
        return best
    else:
        best = float('inf')
        for move in legal_moves(b):
            val = minimax(apply_move(b, move, opponent), d - 1, True, p, nodes)
            best = min(best, val)
        return best

# ==== Alpha-Beta ====
def alphabeta(b: Board, d: int, alpha: int, beta: int, maximizing: bool, p: Player, nodes: List[int]) -> int:
    nodes[0] += 1
    if d == 0 or is_draw(b) or is_winner(b, 'X') or is_winner(b, 'O'):
        return evaluate(b, p)
    opponent = 'O' if p == 'X' else 'X'
    if maximizing:
        value = -float('inf')
        for move in legal_moves(b):
            value = max(value, alphabeta(apply_move(b, move, p), d - 1, alpha, beta, False, p, nodes))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for move in legal_moves(b):
            value = min(value, alphabeta(apply_move(b, move, opponent), d - 1, alpha, beta, True, p, nodes))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

# ==== Depth-Limited Minimax with Heuristic ====
def limited_minimax(b: Board, d: int, maximizing: bool, p: Player, nodes: List[int]) -> int:
    nodes[0] += 1
    if d == 0 or is_draw(b) or is_winner(b, 'X') or is_winner(b, 'O'):
        return simple_heuristic(b, p)
    opponent = 'O' if p == 'X' else 'X'
    if maximizing:
        best = -float('inf')
        for move in legal_moves(b):
            val = limited_minimax(apply_move(b, move, p), d - 1, False, p, nodes)
            best = max(best, val)
        return best
    else:
        best = float('inf')
        for move in legal_moves(b):
            val = limited_minimax(apply_move(b, move, opponent), d - 1, True, p, nodes)
            best = min(best, val)
        return best

# ==== Agent decision wrappers ====
def best_minimax(b: Board, d: int, p: Player) -> tuple[int, int, int]:
    best_score = -float('inf')
    best_move = -1
    nodes = [0]
    start = time.perf_counter_ns()
    for move in legal_moves(b):
        score = minimax(apply_move(b, move, p), d - 1, False, p, nodes)
        if score > best_score:
            best_score = score
            best_move = move
    end = time.perf_counter_ns()
    return best_move, end - start, nodes[0]

def best_alphabeta(b: Board, d: int, p: Player) -> tuple[int, int, int]:
    best_score = -float('inf')
    best_move = -1
    nodes = [0]
    alpha, beta = -float('inf'), float('inf')
    start = time.perf_counter_ns()
    for move in legal_moves(b):
        score = alphabeta(apply_move(b, move, p), d - 1, alpha, beta, False, p, nodes)
        if score > best_score:
            best_score = score
            best_move = move
    end = time.perf_counter_ns()
    return best_move, end - start, nodes[0]

def best_limited(b: Board, d: int, p: Player) -> tuple[int, int, int]:
    best_score = -float('inf')
    best_move = -1
    nodes = [0]
    start = time.perf_counter_ns()
    for move in legal_moves(b):
        score = limited_minimax(apply_move(b, move, p), d - 1, False, p, nodes)
        if score > best_score:
            best_score = score
            best_move = move
    end = time.perf_counter_ns()
    return best_move, end - start, nodes[0]

def best_heuristic(b: Board, d: int, p: Player) -> tuple[int, int, int]:
    best_score = -float('inf')
    best_move = -1
    start = time.perf_counter_ns()
    for move in legal_moves(b):
        new_b = apply_move(b, move, p)
        score = simple_heuristic(new_b, p)
        if score > best_score:
            best_score = score
            best_move = move
    end = time.perf_counter_ns()
    return best_move, end - start, len(legal_moves(b))

def random_opp(b: Board) -> int:
    return random.choice(legal_moves(b))

# ==== Benchmark loop ====
def run_avg_benchmark(fn, label: str):
    results = []
    for depth in range(1, 6):
        total_time = total_nodes = total_moves = 0
        wins = draws = losses = 0
        for _ in range(50):
            board = initial_board()
            current_player = 'X'
            move_times = []
            node_counts = []
            while not is_draw(board) and not is_winner(board, 'X') and not is_winner(board, 'O'):
                if current_player == 'X':
                    move, t, n = fn(board, depth, 'X')
                    move_times.append(t)
                    node_counts.append(n)
                else:
                    move = random_opp(board)
                board = apply_move(board, move, current_player)
                current_player = 'O' if current_player == 'X' else 'X'
            result = evaluate(board, 'X')
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            total_time += sum(move_times)
            total_nodes += sum(node_counts)
            total_moves += len(move_times)
        results.append({
            "depth": depth,
            "avg_time_ns": total_time / total_moves if total_moves else 0,
            "avg_nodes": total_nodes / total_moves if total_moves else 0,
            "win_rate": wins / 50,
            "draw_rate": draws / 50,
            "loss_rate": losses / 50
        })
    df = pd.DataFrame(results)
    df.to_csv(f"{label}_averaged.csv", index=False)
    print(f"✅ Saved: {label}_averaged.csv")

# ==== Run all agents ====
run_avg_benchmark(best_minimax, "minimax")
run_avg_benchmark(best_alphabeta, "alphabeta")
run_avg_benchmark(best_heuristic, "heuristic")
run_avg_benchmark(best_limited, "limited_minimax")
