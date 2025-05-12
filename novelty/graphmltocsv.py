import networkx as nx
import pandas as pd

# Load the .graphml file
graph_path = r"C:\Users\LENOVO\GRAPHRAG\outputt.graphml"
G = nx.read_graphml(graph_path)

# Convert the graph to GraphML string
graphml_str = "\n".join(nx.generate_graphml(G))

# Create a DataFrame with a single row and column
df = pd.DataFrame({'entity_graph': [graphml_str]})

# Save to CSV
output_path = r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_with_clusters.csv"
df.to_csv(output_path, index=False)

print("✅ GraphML successfully converted to 1-row, 1-column CSV.")

