import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, identity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import xml.etree.ElementTree as ET
def group_similar_edge_types(edge_types, similarity_threshold=0.5):
    """Clusters similar edge types based on textual similarity using DBSCAN."""
    unique_edge_descriptions = list(set(edge_types.values()))

    # Convert edge descriptions into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(unique_edge_descriptions)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(X)

    # Ensure similarity values are non-negative
    similarity_matrix = np.clip(similarity_matrix, 0, 1)  # Clip values to [0,1]

    # Convert similarity to distance (higher similarity = lower distance)
    distance_matrix = 1 - similarity_matrix

    # Apply DBSCAN (lets the algorithm decide the number of clusters)
    clustering = DBSCAN(eps=similarity_threshold, min_samples=2, metric="precomputed")
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Map edge descriptions to clusters
    edge_mapping = {}
    for i, desc in enumerate(unique_edge_descriptions):
        if cluster_labels[i] == -1:  # Noise (not clustered)
            edge_mapping[desc] = f"unique_relation_{i}"  # Assign a unique name
        else:
            edge_mapping[desc] = f"relation_{cluster_labels[i]}"


    return edge_mapping

def preprocess_heterogeneous_graph(graphml_path, similarity_threshold=0.5):
    # Load the graph
    graph = nx.read_graphml(graphml_path)
    tree = ET.parse(graphml_path)
    root = tree.getroot()  # ✅ Fix: Define root before using it
    
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}  # GraphML namespace

    # Extract node attributes
    node_types = {node: data.get("type", "default") for node, data in graph.nodes(data=True)}
    node_features={node:data for node,data in graph.nodes(data=True)}
    node_class = {}
    for node in root.findall(".//g:node", ns):
        entity_data = node.find(".//g:data[@key='d3']", ns)  # Extract entity type (d3)
        node_class[node.get("id")] = entity_data.text.strip() if entity_data is not None and entity_data.text.strip() else "unknown"

    # Extract edge attributes
    edge_types = {(u, v): data.get("description", "default") for u, v, data in graph.edges(data=True)}
    
    # Cluster edge types
    edge_mapping = group_similar_edge_types(edge_types, similarity_threshold)
    
    # Apply the mapping to edges
    processed_edges = {(u, v): edge_mapping[data.get("description", "default")] for u, v, data in graph.edges(data=True)}
    edge_counts = Counter(processed_edges.values())
    #print("Edge Type Counts:", edge_counts)
    # Convert edge types to indices
    edge_type_encoder = LabelEncoder()
    encoded_edge_types = edge_type_encoder.fit_transform(list(set(edge_mapping.values())))
    final_edge_mapping = {etype: encoded_type for etype, encoded_type in zip(set(edge_mapping.values()), encoded_edge_types)}
    
    # Construct adjacency matrices

        # Map nodes to indices
    node_list = list(graph.nodes)
    node_idx_map = {node: idx for idx, node in enumerate(node_list)}
    adjacency_matrices = {}
    for edge_type in set(final_edge_mapping.values()):
        edges_of_type = [(u, v) for (u, v), etype in processed_edges.items() if etype in final_edge_mapping]
    
        print(f"Edge Type: {edge_type}, Number of Edges: {len(edges_of_type)}")  # Debugging
    
        if edges_of_type:
            rows, cols = zip(*[(node_idx_map[u], node_idx_map[v]) for u, v in edges_of_type])
            adj_matrix = coo_matrix(
                (np.ones(len(rows)), (rows, cols)), 
                shape=(len(node_list), len(node_list))
            )
            adj_matrix_normalized = normalize_adjacency_matrix(adj_matrix)
            adjacency_matrices[edge_type] = adj_matrix_normalized
    
    return {
        "graph": graph,
        "node_features": node_features,
        "node_types": node_types,
        "edge_types": final_edge_mapping,
        "adjacency_matrices": adjacency_matrices,
        "labels": node_class, 
    }

def normalize_adjacency_matrix(adj):
    """Normalize adjacency matrix using symmetric normalization."""
    adj = adj + identity(adj.shape[0])  # Add self-loops
    degrees = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
    d_mat_inv_sqrt = coo_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))))
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# Path to your GraphML file
graphml_path = r"C:\Users\LENOVO\GRAPHRAG\MIE\create_base_extracted_entities.graphml"

# Preprocess the graph
processed_graph = preprocess_heterogeneous_graph(graphml_path, similarity_threshold=0.5)

# Display summary
print("Graph Preprocessing Complete!")
print(f"Reduced Edge Types: {len(set(processed_graph['edge_types'].values()))}")
print(f"Number of Node Types: {len(set(processed_graph['node_types'].values()))}")
print(f"Number of Edge Types: {len(set(processed_graph['edge_types'].values()))}")
print(f"Adjacency Matrices Created: {len(processed_graph['adjacency_matrices'])}")
print(f"Labels Extracted: {len(processed_graph['labels'])} nodes labeled")
print(f"Sample Labels: {list(processed_graph['labels'].items())[:10]}")

