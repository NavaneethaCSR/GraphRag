import os
import pickle
from node2vec import Node2Vec
from graphpreprocessing import processed_graph
import networkx as nx

def compute_node2vec_embeddings(graph, model_cache=r"C:\Users\LENOVO\GRAPHRAG\MIE\node2vec.pkl"):
    # Convert graph to undirected if needed
    G = graph.to_undirected() if not nx.is_directed(graph) else graph

    # ✅ Check if cached model exists to avoid recomputation
    if os.path.exists(model_cache):
        print("Loading cached Node2Vec model...")
        with open(model_cache, "rb") as f:
            model = pickle.load(f)  # ✅ Load trained model
    else:
        print("Generating random walks & training model...")
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # ✅ Cache the trained model
        with open(model_cache, "wb") as f:
            pickle.dump(model, f)

    # ✅ Extract node embeddings
    node_embeddings = {node: model.wv[str(node)] for node in G.nodes()}
    return node_embeddings

# ✅ Compute embeddings (loads cached model if available)
node_embeddings = compute_node2vec_embeddings(processed_graph["graph"])
