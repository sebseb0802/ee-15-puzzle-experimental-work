import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import time

# STEP 1: Load GTFS files
stops = pd.read_csv("stops.txt")
routes = pd.read_csv("routes.txt")
trips = pd.read_csv("trips.txt")
stop_times = pd.read_csv("stop_times.txt")

# STEP 2: Filter tram routes only (route_type == 0)
tram_routes = routes[routes["route_type"] == 0]

# STEP 3: Select a controlled number of tram routes
n = 10000  # <-- change this to increase input size (e.g. 10, 20, ...)
subset_route_ids = tram_routes["route_id"].unique()[:n]

# STEP 4: Filter trips to only those in the selected tram routes
subset_trips = trips[trips["route_id"].isin(subset_route_ids)]

# STEP 5: Drop duplicate trips per route (keep only one trip per route)
subset_trips = subset_trips.drop_duplicates(subset="route_id")

# STEP 6: Filter stop_times and remove redundant entries
subset_stop_times = stop_times[stop_times["trip_id"].isin(subset_trips["trip_id"])]
subset_stop_times = subset_stop_times.drop_duplicates(subset=["trip_id", "stop_id", "stop_sequence"])

# STEP 7: Build stop coordinate lookup from stops.txt
stop_coords = {
   row["stop_id"]: (row["stop_lat"], row["stop_lon"])
   for _, row in stops.iterrows()
}

# STEP 8: Build directed graph
G = nx.DiGraph()

# Add nodes
for stop_id, (lat, lon) in stop_coords.items():
   G.add_node(stop_id, pos=(lat, lon))

# Add edges using ordered stop_times
for trip_id, group in subset_stop_times.groupby("trip_id"):
   group = group.sort_values("stop_sequence")
   prev_row = None
   for _, row in group.iterrows():
       if prev_row is not None:
           u = prev_row["stop_id"]
           v = row["stop_id"]
           coord_u = stop_coords.get(u)
           coord_v = stop_coords.get(v)
           if coord_u and coord_v and u != v:
               distance = geodesic(coord_u, coord_v).meters
               G.add_edge(u, v, weight=distance)
               G.add_edge(v, u, weight=distance)
       prev_row = row

# STEP 9: Clean the graph
G.remove_nodes_from(list(nx.isolates(G)))
if not nx.is_connected(G.to_undirected()):
   largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
   G = G.subgraph(largest_cc).copy()

# STEP 10: Visualize and save graph
pos = {node: data["pos"] for node, data in G.nodes(data=True)}
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=10, edge_color="gray", with_labels=False)
plt.savefig("vienna_tram_subset.png", dpi=300, bbox_inches="tight")
plt.close()

# STEP 11: Run Dijkstra
nodes = list(G.nodes)
max_dist = 0
best_pair = (None, None)
for i in range(len(nodes)):
   for j in range(i + 1, len(nodes)):
       coord_i = G.nodes[nodes[i]]["pos"]
       coord_j = G.nodes[nodes[j]]["pos"]
       dist = geodesic(coord_i, coord_j).meters
       if dist > max_dist:
           max_dist = dist
           best_pair = (nodes[i], nodes[j])

start_node, end_node = best_pair

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

start_time = time.perf_counter()
path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight", method="dijkstra")
duration = time.perf_counter() - start_time
print(f"Shortest path from {start_node} to {end_node}:")
print(path)
print(f"Path length: {nx.path_weight(G, path, weight='weight'):.2f} meters")
print(f"Time taken: {duration:.6f} seconds")