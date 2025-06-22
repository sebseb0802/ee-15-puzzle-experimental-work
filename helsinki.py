import os
import osmnx as ox
import networkx as nx
import random
import heapq
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# === Config ===
PLACE_NAME = "London Borough of Wandsworth, UK"
MAX_DEPTH = 20
RUNS_PER_PAIR = 5
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
random.seed(42)

# === Download and simplify walkable graph ===
print("Downloading and simplifying pedestrian network...")
G_raw = ox.graph_from_place(PLACE_NAME, network_type="walk", simplify=False)
G = ox.simplify_graph(G_raw)

# Convert to undirected for connected components
G_u = G.to_undirected()
G_u = G_u.subgraph(max(nx.connected_components(G_u), key=len)).copy()
G = nx.convert_node_labels_to_integers(G_u)
G_proj = ox.project_graph(G)  # use meter-based CRS

# === Heuristics ===
positions = {n: (d['x'], d['y']) for n, d in G_proj.nodes(data=True)}

def euclidean(u, v):
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def overestimated(u, v):
    return 1.5 * euclidean(u, v)

def zero(u, v): return 0

heuristics = {
    "Zero": zero,
    "Euclidean": euclidean,
    "Overestimated Euclidean": overestimated
}

# === A* Search ===
def a_star(start, goal, heuristic_fn):
    open_set = [(heuristic_fn(start, goal), 0, start)]
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return cost, len(visited)
        if current in visited:
            continue
        visited.add(current)

        for _, neighbor, data in G_proj.edges(current, data=True):
            dist = data.get("length", 1.0)
            new_cost = g_score[current] + dist
            if neighbor not in g_score or new_cost < g_score[neighbor]:
                g_score[neighbor] = new_cost
                priority = new_cost + heuristic_fn(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
    return float('inf'), len(visited)

def find_node_pairs_by_depth(max_depth=20):
    print("Finding node pairs with exact edge depth...")
    pairs_by_depth = {}
    nodes = list(G_proj.nodes)

    while len(pairs_by_depth) < max_depth:
        u = random.choice(nodes)
        try:
            lengths = nx.single_source_shortest_path_length(G_proj, u, cutoff=max_depth)
            for v, depth in lengths.items():
                if u != v and 1 <= depth <= max_depth and depth not in pairs_by_depth:
                    pairs_by_depth[depth] = (u, v)
                    if len(pairs_by_depth) == max_depth:
                        break
        except:
            continue
    return pairs_by_depth

# === Benchmarking ===
def benchmark():
    print("Benchmarking A* heuristics from depth 1 to", MAX_DEPTH)
    node_pairs = find_node_pairs_by_depth(MAX_DEPTH)

    for name, hfn in heuristics.items():
        print(f"→ {name}")
        records = []
        for depth in tqdm(range(1, MAX_DEPTH + 1)):
            start, goal = node_pairs[depth]
            times, nodes, ratios = [], [], []

            optimal_cost, _ = a_star(start, goal, euclidean)

            for _ in range(RUNS_PER_PAIR):
                t0 = time.perf_counter_ns()
                cost, expanded = a_star(start, goal, hfn)
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)
                nodes.append(expanded)
                ratio = cost / optimal_cost if optimal_cost > 0 else 1.0
                ratios.append(ratio)

            records.append({
                "depth_edges": depth,
                "avg_time_ns": np.mean(times),
                "avg_nodes_expanded": np.mean(nodes),
                "avg_optimality_ratio": np.mean(ratios)
            })

        df = pd.DataFrame(records)
        filename = f"{DATA_DIR}/battersea_depth_{name.lower().replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")

benchmark()
