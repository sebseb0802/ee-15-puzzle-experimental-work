import osmnx as ox
import networkx as nx

north, south, east, west = 47.4000, 47.3300, 13.8000, 13.6200

custom_filter = (
    '["piste:type"~"downhill"]["aerialway]'
)

G = ox.graph_from_bbox(
    north=north,
    south=south,
    east=east,
    west=west,
    custom_filter=custom_filter,
    simplify=True,
    retain_all=True,
    network_type="all-private"
)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Directed: {nx.is_directed(G)}")

ox.plot_graph(G, node_size=10, edge_linewidth=0.8, bgcolor='white')