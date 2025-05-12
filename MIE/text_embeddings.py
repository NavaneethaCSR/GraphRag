from sentence_transformers import SentenceTransformer
import numpy as np
from graphpreprocessing import processed_graph 
def extract_text_features(graph, node_features, text_attribute="description"):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight BERT model
    
    text_embeddings = {}
    for node, attrs in node_features.items():
        text = attrs.get(text_attribute, "")  # Extract description or default to empty string
        embedding = model.encode(text) if text else np.zeros(384)  # BERT output is 384-dim
        text_embeddings[node] = embedding
    
    return text_embeddings

# Extract text features
text_features = extract_text_features(processed_graph["graph"], processed_graph["node_features"])
