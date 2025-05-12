import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg
import networkx as nx
# Load node embeddings

graph_path = r"C:\Users\LENOVO\GRAPHRAG\outputt11 (2).graphml"
G = nx.read_graphml(graph_path)
embeddings_df = pd.read_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings.csv", index_col=False)

# Extract node IDs (preserving order)
node_ids = list(G.nodes)
embeddings_df.insert(0, 'id', node_ids)
embeddings_df.to_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings.csv", index=False)

print("✅ Node IDs from GraphML successfully added to embeddings.")
ids = embeddings_df["id"].values  # Replace 'node_id' if your ID column has a different name
embeddings = embeddings_df.drop(columns=["id"]).values

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Build graph using Top-K similar nodes
top_k = 5
edges = []
for i in range(len(ids)):
    sims = list(enumerate(similarity_matrix[i]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_k+1]  # skip self
    for j, sim in sims:
        edges.append((i, j, sim))

# Create igraph graph
g = ig.Graph()
g.add_vertices(len(ids))
g.add_edges([(src, dst) for src, dst, _ in edges])
g.es["weight"] = [w for _, _, w in edges]

# Run Leiden algorithm
partition = leidenalg.find_partition(
    g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"]
)
clusters = partition.membership

# Attach cluster info to DataFrame
embeddings_df["cluster"] = clusters

# Save updated CSV
embeddings_df.to_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings(3).csv", index=False)
