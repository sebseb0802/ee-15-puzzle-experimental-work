import pandas as pd
import networkx as nx
import pyproj
import os
import random
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from math import sqrt
import numpy as np

# === Setup ===
os.makedirs("plots/paths", exist_ok=True)
random.seed(42)

# === Load and preprocess data ===
bus_seq = pd.read_csv("bus-sequences.csv")
bus_seq = bus_seq[['Route', 'Sequence', 'Stop_Name', 'Location_Easting', 'Location_Northing']].dropna()
bus_seq = bus_seq.sort_values(by=['Route', 'Sequence'])

proj = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
bus_seq[['lon', 'lat']] = bus_seq.apply(
    lambda row: proj.transform(row['Location_Easting'], row['Location_Northing']),
    axis=1, result_type='expand'
)

stop_map = {}
for _, row in bus_seq.iterrows():
    name = row['Stop_Name']
    if name not in stop_map:
        stop_map[name] = {'lat': row['lat'], 'lon': row['lon']}

# === Distance and Heuristic ===
def euclidean(coord1, coord2):
    return sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def euclidean_heuristic(n1, n2, pos):
    return euclidean(pos[n1], pos[n2])

# === Build Graph ===
def build_graph():
    G = nx.Graph()
    for name, coord in stop_map.items():
        G.add_node(name, pos=(coord['lat'], coord['lon']))
    for route, group in bus_seq.groupby('Route'):
        stops = group['Stop_Name'].tolist()
        for i in range(len(stops) - 1):
            u, v = stops[i], stops[i + 1]
            if u != v and u in stop_map and v in stop_map:
                c1 = (stop_map[u]['lat'], stop_map[u]['lon'])
                c2 = (stop_map[v]['lat'], stop_map[v]['lon'])
                dist = euclidean(c1, c2)
                G.add_edge(u, v, weight=dist)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G

# === Draw path ===
def draw_path(G, path, filename):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, node_size=5, edge_color='lightgray', with_labels=False)
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', node_size=20)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === Plot depth heatmap ===
def plot_depth_heatmap(G, start_node, pos, depths, out_path):
    colors = [depths.get(n, -1) for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(12, 10))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=colors, cmap=viridis,
        node_size=10, ax=ax, vmin=1, vmax=max(depths.values())
    )
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.5)
    cbar = fig.colorbar(nodes, ax=ax, label='Graph Depth from START_STOP')
    ax.set_title(f"Reachability Heatmap from {start_node}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# === Main logic ===
def visualize_paths_and_depths():
    G = build_graph()
    pos = nx.get_node_attributes(G, 'pos')

    # Find best START_STOP node based on max reachable depth and spread
    print("Finding best START_STOP node...")
    best_node = None
    best_score = -1
    for node in random.sample(list(G.nodes), 200):
        depths = nx.single_source_shortest_path_length(G, node, cutoff=20)
        if len(depths) < 20:
            continue
        score = max(depths.values())
        if score > best_score:
            best_score = score
            best_node = node

    if not best_node:
        raise Exception("Failed to find a suitable START_STOP node with good depth spread.")

    START_STOP = best_node
    print(f"Selected central START_STOP: {START_STOP} (max depth {best_score})")

    depths = nx.single_source_shortest_path_length(G, START_STOP, cutoff=20)
    plot_depth_heatmap(G, START_STOP, pos, depths, "plots/paths/depth_heatmap.png")

    print("Drawing one path per depth...")
    depth_bins = {d: [] for d in range(1, 21)}
    for node, d in depths.items():
        if 1 <= d <= 20:
            depth_bins[d].append(node)

    for depth in range(1, 21):
        candidates = depth_bins[depth]
        if not candidates:
            print(f"Depth {depth}: No nodes.")
            continue
        target = random.choice(candidates)
        try:
            path = nx.astar_path(G, START_STOP, target, heuristic=lambda u, v: euclidean_heuristic(u, v, pos), weight='weight')
            print(f"Depth {depth}: {START_STOP} → {target}, path length = {len(path)-1}")
            draw_path(G, path, f"plots/paths/depth_{depth}.png")
        except nx.NetworkXNoPath:
            print(f"Depth {depth}: No path to {target}")

visualize_paths_and_depths()
