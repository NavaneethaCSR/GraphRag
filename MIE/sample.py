import networkx as nx

# Load the uploaded GraphML file
graph_path = r"C:\Users\LENOVO\GRAPHRAG\outputt11 (2).graphml"
G = nx.read_graphml(graph_path)

# Count the number of nodes
num_nodes = G.number_of_nodes()
print(num_nodes)
