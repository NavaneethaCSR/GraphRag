import torch
import pandas as pd



# Load the tensor from the .pth file
tensor = torch.load((r'C:\Users\LENOVO\GRAPHRAG\node_embeddings(2).pth'), map_location='cpu')

# Convert the tensor to a NumPy array and then to a DataFrame
df = pd.DataFrame(tensor.numpy())

# Save the DataFrame to a CSV file
df.to_csv(r'C:\Users\LENOVO\GRAPHRAG\node_embeddings.csv', index=False)

print("Conversion complete: node_embeddings.csv")
