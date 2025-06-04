import osmnx as ox
import networkx as nx
import matplotlib
import random

from math import radians, cos, sin, sqrt, atan2
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1) * cos(φ2) * sin(dλ/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

print(ox.__version__)

G = ox.graph_from_xml("Graph.osm", simplify=False)
G = ox.project_graph(G)
print(type(G))

ox.simplification.consolidate_intersections(G)


edges_to_remove = [
    (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
    if data.get("reversed") is True
]
G.remove_edges_from(edges_to_remove)

for u, v, k, data in G.edges(data=True, keys=True):
    y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
    y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
    dist = haversine_distance(y1, x1, y2, x2)

    if data.get("aerialway"):
        data["weight"] = 0.1
    else:
        difficulty_tag = data.get("piste:difficulty")
        if difficulty_tag == "easy":
            difficulty = 1
        elif difficulty_tag == "intermediate":
            difficulty = 3
        elif difficulty_tag == "advanced":
            difficulty = 5
        else:
            difficulty = 0

        data["weight"] = dist + difficulty

for u, v, k, data in G.edges(data=True, keys=True):
    print(data)
    if data.get("aerialway"):
        print(f"Lift from {u} to {v}, oneway: {data.get('oneway')}")

nodes = list(G.nodes)
start_node = 367732495
end_node = 367733299
print(f"Start: {start_node}, End: {end_node}")

print("Start degree: ", G.degree[start_node])
print("End degree: ", G.degree[end_node])

components = list(nx.connected_components(G.to_undirected()))
start_component = [c for c in components if start_node in c]
end_component = [c for c in components if end_node in c]
print("Start and end in same component:", start_component == end_component)

path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight", method="dijkstra")

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Directed: {nx.is_directed(G)}")

ox.plot_graph_route(
    G=G, 
    route=path, 
    route_color="b", 
    route_linewidth=3, 
    bgcolor="red",
    save=True, 
    filepath="shortest_path.png"
)

ox.plot_graph(G, node_size=10, edge_linewidth=1, bgcolor='red', show=False, save=True, filepath="graph.png")