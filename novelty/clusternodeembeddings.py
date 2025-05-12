from sklearn.cluster import KMeans
import pandas as pd

embeddings = pd.read_csv(r'C:\Users\LENOVO\GRAPHRAG\node_embeddings.csv', index_col=0)
kmeans = KMeans(n_clusters=5, random_state=42)
embeddings['cluster'] = kmeans.fit_predict(embeddings)

# Save or inspect
embeddings.to_csv(r"C:\Users\LENOVO\GRAPHRAG\node_embeddings_with_clusters.csv")
