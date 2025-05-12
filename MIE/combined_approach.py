import numpy as np
from sklearn.preprocessing import StandardScaler
from text_embeddings import *
from community_embeddings import *
from node_embeddings import *


def combine_features(graph, node_embeddings, text_features, community_one_hot):
    # Sort nodes for consistent order
    sorted_nodes = sorted(graph.nodes())

    # Convert dictionaries to matrices
    node_emb_matrix = np.array([node_embeddings[node] for node in sorted_nodes])
    text_emb_matrix = np.array([text_features[node] for node in sorted_nodes])

    # Normalize numerical features
    scaler = StandardScaler()
    node_emb_matrix = scaler.fit_transform(node_emb_matrix)
    text_emb_matrix = scaler.fit_transform(text_emb_matrix)

    # Concatenate all features
    final_feature_matrix = np.hstack([node_emb_matrix, text_emb_matrix, community_one_hot])

    return final_feature_matrix

# Combine all features
final_feature_matrix = combine_features(processed_graph["graph"], node_embeddings, text_features, community_one_hot)
