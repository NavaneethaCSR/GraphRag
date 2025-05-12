import pandas as pd
import networkx as nx


import networkx as nx
import pandas as pd

# # Load the graph
# graph_path = r"C:\Users\LENOVO\GRAPHRAG\ragtest3\output\create_base_extracted_entities.graphml"
# G = nx.read_graphml(graph_path)

# # Extract node IDs (preserving order)
# node_ids = list(G.nodes)

# # Load embeddings (clusters already assigned)
# embeddings_df = pd.read_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_clusters.csv", index_col=False)

# # Validate lengths match
# if len(node_ids) != len(embeddings_df):
#     raise ValueError("Mismatch: number of nodes in GraphML vs rows in embeddings CSV")

# # Add node_id as first column
# embeddings_df.insert(0, 'node_id', node_ids)

# # Save the updated CSV
# embeddings_df.to_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_clusters.csv", index=False)

# print("✅ Node IDs from GraphML successfully added to embeddings.")



# # Step 3: Load embeddings with clusters
# embedding_path = r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_with_clusters.csv"
# df = pd.read_csv(embedding_path, index_col=False)

# # Step 4: Check if number of nodes matches
# assert len(node_ids) == len(df), "Mismatch between number of nodes and embeddings!"

# # Step 5: Assign correct node IDs
# df['node_id'] = node_ids
# df.set_index('node_id', inplace=True)

# # Step 6: Add cluster info to the graph
# for node in G.nodes():
#     if node in df.index:
#         G.nodes[node]['kmeans_cluster'] = int(df.loc[node, 'cluster'])
#     else:
#         G.nodes[node]['kmeans_cluster'] = -1

# # Step 7: Save the updated graph
# nx.write_graphml(G, r"C:\Users\LENOVO\GRAPHRAG\outputt_with_fixed_clusters.graphml")
# print("✅ Clusters aligned and saved as outputt_with_fixed_clusters.graphml")

# import pandas as pd
# import networkx as nx

# # Load graph
# G = nx.read_graphml(r"C:\Users\LENOVO\GRAPHRAG\MIE\output_graph.graphml")

# # Load your CSV where the first column is node ID (e.g. WILLY WONKA)
# df = pd.read_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_with_clusters.csv", sep=",", header=None)


# # Create mapping
# df.columns = ['node_id'] + [f'emb_{i}' for i in range(df.shape[1] - 2)] + ['cluster']
# df.set_index('node_id', inplace=True)

# # Assign clusters to graph nodes using node IDs
# for node in G.nodes():
#     if node in df.index:
#         G.nodes[node]['d3'] = int(df.loc[node, 'cluster'])  # Preserve cluster
#     else:
#         G.nodes[node]['d3'] = -1  # Unassigned

# # Save the updated GraphML
# nx.write_graphml(G, r"C:\Users\LENOVO\GRAPHRAG\outputt_with_preserved_kmeans.graphml")

import networkx as nx

# Load the preserved GraphML file
graphml_path = r"C:\Users\LENOVO\GRAPHRAG\outputt1.graphml"  # Update with your actual file
G = nx.read_graphml(graphml_path)

# Add degree to each node
for node in G.nodes():
    G.nodes[node]['degree'] = G.degree(node)

# Save the updated graph
nx.write_graphml(G, r"C:\Users\LENOVO\GRAPHRAG\new_outputt1.graphml")

print("Degrees added and graph saved as 'outputt_with_cluster_and_degree.graphml'")
