# import networkx as nx
# import pandas as pd

# # Load your GraphML file
# G = nx.read_graphml(r"C:\Users\LENOVO\GRAPHRAG\outputt.graphml")

# # Extract labels from GraphML nodes (assumed to be stored under 'label')
# node_labels = []
# for node_id, attrs in G.nodes(data=True):
#     label = attrs.get("label", node_id)  # fallback to node_id if label doesn't exist
#     node_labels.append(label)

# # Load your CSV file
# df = pd.read_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_with_clusters.csv")

# # Add the labels as a new column at the start (node_id column)
# df.insert(0, "node_id", node_labels)

# # Save the updated CSV
# df.to_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_with_clusters.csv", index=False)


import networkx as nx
import pandas as pd

# Load GraphML graph
graph_path = r"C:\Users\LENOVO\GRAPHRAG\outputt11 (2).graphml"
G = nx.read_graphml(graph_path)

# Load embeddings with clusters
cluster_df = pd.read_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings(3).csv")

# Create a mapping: node_id -> cluster
cluster_map = dict(zip(cluster_df["id"], cluster_df["cluster"]))

# Assign cluster to each node
for node in G.nodes():
    G.nodes[node]["cluster"] = int(cluster_map.get(node, -1))  # Default -1 if not found

# Save updated graph
output_path = r"C:\Users\LENOVO\GRAPHRAG\outputt11 (2).graphml"
nx.write_graphml(G, output_path)

print("✅ Clustered GraphML saved.")


