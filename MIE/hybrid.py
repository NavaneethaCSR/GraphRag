import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_semantic_graph(embedding_csv_path: str, n_neighbors: int = 10) -> nx.Graph:
    df = pd.read_csv(embedding_csv_path)

    # Drop rows where 'id' is missing
    df = df.dropna(subset=['id'])

    node_ids = df['id'].tolist()
    embeddings = df.drop(columns=['id', 'cluster']).values

    knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(node_ids) - 1), metric='cosine')
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)

    semantic_graph = nx.Graph()
    for i, neighbors in enumerate(indices):
        for offset, j in enumerate(neighbors):
            if i != j:
                weight = 1 - distances[i][offset]  # cosine similarity
                semantic_graph.add_edge(node_ids[i], node_ids[j], weight=weight)
    return semantic_graph


def merge_graphs(original_graph_path: str, semantic_graph: nx.Graph) -> nx.Graph:
    original_graph = nx.read_graphml(original_graph_path)
    merged_graph = nx.Graph()

    # Add original edges
    merged_graph.add_edges_from(original_graph.edges(data=True))
    # Add semantic edges (ignore duplicates or merge weights if needed)
    for u, v, data in semantic_graph.edges(data=True):
        if merged_graph.has_edge(u, v):
            continue  # or handle merging weights here
        merged_graph.add_edge(u, v, **data)

    return merged_graph

if __name__ == "__main__":
    embedding_csv = r"C:\Users\LENOVO\GRAPHRAG\node_embeddings(2) with clusters.csv"
    original_graphml = r"C:\Users\LENOVO\GRAPHRAG\outputt1.graphml"
    output_path = r"C:\Users\LENOVO\GRAPHRAG\hybrid.graphml"

    print("Building semantic (embedding-based) graph...")
    semantic_g = build_semantic_graph(embedding_csv, n_neighbors=10)

    print("Merging with original graph structure...")
    hybrid_g = merge_graphs(original_graphml, semantic_g)

    print(f"Saving hybrid graph to {output_path}...")
    nx.write_graphml(hybrid_g, output_path)

    print("Done.")
