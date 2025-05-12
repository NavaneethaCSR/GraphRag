import community.community_louvain as community # python-louvain
import numpy as np
from graphpreprocessing import processed_graph 
def detect_communities(graph):
    partition = community.best_partition(graph)  # Compute Louvain communities
    return partition

# Compute community membership
community_labels = detect_communities(processed_graph["graph"])

# Convert to one-hot encoding
num_communities = len(set(community_labels.values()))
community_one_hot = np.zeros((len(community_labels), num_communities))
for i, node in enumerate(processed_graph["graph"].nodes()):
    community_one_hot[i, community_labels[node]] = 1
