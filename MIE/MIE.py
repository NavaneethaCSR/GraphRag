import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from graphpreprocessing import processed_graph  # Custom module for graph data
from combined_approach import *
from torch.optim.lr_scheduler import OneCycleLR
# Feature Preprocessing
def preprocess_features(x):
    """ Standardize features: mean = 0, std = 1 """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)  # ✅ Convert to PyTorch tensor

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6  # ✅ Avoid division by zero

    return (x - mean) / std



# Data Augmentation Functions
def add_noise(x, noise_level=0.1):
    """ Add Gaussian noise for robustness """
    noise = torch.randn_like(x) * noise_level
    return x + noise

def feature_dropout(x, dropout_rate=0.2):
    """ Randomly set some features to zero """
    mask = (torch.rand_like(x) > dropout_rate).float()
    return x * mask

def jitter(x, scale=0.02):
    """ Small perturbations to feature values """
    return x + (torch.rand_like(x) - 0.5) * scale

def graphmix(features, adj, alpha=1.0):
    lam = np.random.beta(alpha, alpha)

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)

    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj.toarray(), dtype=torch.float32)

    # Ensure everything is on same device
    device = features.device
    adj = adj.to(device)

    neighbors = (adj > 0).float()
    neighbors.fill_diagonal_(0)

    row_sums = neighbors.sum(dim=1)
    valid_indices = (row_sums > 0).nonzero(as_tuple=False).squeeze()

    if valid_indices.numel() == 0:
        return features

    idx = random.choice(valid_indices.tolist())
    probs = neighbors[idx] / neighbors[idx].sum()
    rand_idx = torch.multinomial(probs, 1).item()

    mixed_features = lam * features[idx] + (1 - lam) * features[rand_idx]
    features[idx] = mixed_features
    features = features.clone().detach()  # Ensure it doesn't mess up gradients
    assert not torch.isnan(features).any(), "graphmix(): NaN detected in features"
    assert not torch.isinf(features).any(), "graphmix(): Inf detected in features"

    return features



# Initialize model weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        if hasattr(m, 'lin') and m.lin is not None:
            nn.init.xavier_uniform_(m.lin.weight)
            if m.lin.bias is not None:
                nn.init.zeros_(m.lin.bias)

def global_summary(embeddings):
    """Compute the global summary vector using mean pooling."""
    summary = torch.mean(torch.stack(embeddings), dim=0)

    # ✅ Check for NaN values before returning
    assert not torch.isnan(summary).any(), "Global summary vector contains NaN values!"
    assert not torch.isinf(summary).any(), "Global summary vector contains Inf values!"

    return summary

# Relation-Specific GNN
class RelationSpecificGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(RelationSpecificGNN, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)  # Added dropout for regularization

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        x = self.dropout(x)  # Apply dropout
        return x

# Semantic Attention for Fusion
# Enhanced Relation-aware Semantic Attention Fusion
class SemanticAttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(SemanticAttentionFusion, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)  # Scalar score per relation
        )

    def forward(self, embeddings):
        """
        embeddings: List of [num_nodes x hidden_dim] tensors for each relation
        """
        # Stack to get [num_nodes, num_relations, hidden_dim]
        stacked = torch.stack(embeddings, dim=1)  

        # Compute attention scores: [num_nodes, num_relations, 1]
        scores = self.projection(stacked)

        # Normalize across relations: [num_nodes, num_relations, 1]
        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum: [num_nodes, hidden_dim]
        fused_output = torch.sum(attention_weights * stacked, dim=1)

        return fused_output


# Main Model: MIEHetGRL
class MIEHetGRL(nn.Module):
    def __init__(self, num_relations, input_dim, hidden_dim, dropout):
        super(MIEHetGRL, self).__init__()
        self.relation_gnns = nn.ModuleList([
            RelationSpecificGNN(input_dim, hidden_dim, dropout) for _ in range(num_relations)
        ])
        self.semantic_attention = SemanticAttentionFusion(hidden_dim)

    def forward(self, x, adjacency_matrices):
        embeddings = []
        for i, adj_matrix in enumerate(adjacency_matrices):
            edge_index, _ = from_scipy_sparse_matrix(adj_matrix.tocoo())
            edge_index = edge_index.to(x.device)
            embeddings.append(self.relation_gnns[i](x, edge_index))
        fused_embeddings = self.semantic_attention(embeddings)
         # Compute the global summary vector
        global_summary_vector = global_summary(embeddings)
        return fused_embeddings,global_summary_vector

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.14613665736179657):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, node_embed, summary_embed, corrupted_embed, corrupted_summary):
        # Normalize all embeddings
        node_embed = F.normalize(node_embed, dim=1)
        corrupted_embed = F.normalize(corrupted_embed, dim=1)
        summary_embed = F.normalize(summary_embed, dim=0)  # global
        corrupted_summary = F.normalize(corrupted_summary, dim=0)

        # Positive similarity: node vs correct summary
        pos_sim = torch.matmul(node_embed, summary_embed.T) / self.temperature

        # Negative similarity: node vs corrupted summary
        neg_sim = torch.matmul(node_embed, corrupted_summary.T) / self.temperature

        # Joint contrastive loss (InfoNCE)
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [N, 2]
        labels = torch.zeros(node_embed.size(0), dtype=torch.long).to(node_embed.device)  # positives at index 0

        loss = F.cross_entropy(logits, labels)

        assert not torch.isnan(loss).any(), "Joint contrastive loss contains NaN!"
        return loss


# Hyperparameters
input_dim = final_feature_matrix.shape[1]  
hidden_dim = 59
dropout = 0.25050350524150783# Regularization: Dropout rate
weight_decay = 1e-4 # Regularization: Weight decay (L2 penalty)
learning_rate = 0.0014335547775991358 # Adjusted learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not isinstance(final_feature_matrix, torch.Tensor):
    final_feature_matrix = torch.tensor(final_feature_matrix, dtype=torch.float32)


# Prepare data
x = preprocess_features(final_feature_matrix)  # Standardization
assert not torch.isnan(x).any(), "NaN in features"
assert not torch.isinf(x).any(), "Inf in features"
x = F.normalize(x, p=2, dim=1)  # Normalization
x = x.to(device)
assert not torch.isnan(x).any(), "Node features contain NaN values!"
assert not torch.isinf(x).any(), "Node features contain Inf values!"

num_nodes = x.size(0)
adjacency_matrices = list(processed_graph["adjacency_matrices"].values())
subset_adjacency_matrices =adjacency_matrices
num_relations = len(adjacency_matrices)

# Model, optimizer, and scheduler
model = MIEHetGRL(num_relations, input_dim, hidden_dim, dropout).to(device)
model.apply(initialize_weights)
batch_size=16
epochs=100
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=1 , epochs=epochs)
contrastive_loss = ContrastiveLoss()

# Training loop
if __name__ == "__main__":
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

         # Apply Data Augmentation
        x_aug = add_noise(x, 0.1)  # Gaussian noise
        x_aug = feature_dropout(x_aug, 0.2)  # Dropout
        x_aug = jitter(x_aug, 0.02)  # Jitter
        x_aug = graphmix(x_aug, adjacency_matrices[0])
        x_aug = x_aug.to(device) 
        assert x_aug.shape == x.shape,  "Shape mismatch in mixup operation!"
   
        assert not torch.isnan(x_aug).any(), "NaN in features after augmentation"
        assert not torch.isinf(x_aug).any(), "Inf in features after augmentation"
        # Forward pass
        node_embeddings,global_summary_vector = model(x_aug, subset_adjacency_matrices)
        assert not torch.isnan(node_embeddings).any(), "Node embeddings contain NaN values!"
        assert not torch.isinf(node_embeddings).any(), "Node embeddings contain Inf values!"
        
        # ✅ Check for NaN/Inf in global summary vector
        assert not torch.isnan(global_summary_vector).any(), "Global summary vector contains NaN values!"
        assert not torch.isinf(global_summary_vector).any(), "Global summary vector contains Inf values!"

        # Contrastive learning
       
        corrupted_x = x_aug[torch.randperm(x_aug.size(0))]
        corrupted_embeddings, corrupted_summary = model(corrupted_x, subset_adjacency_matrices)

        assert not torch.isnan(corrupted_embeddings).any(), "Corrupted embeddings contain NaN values!"
        assert not torch.isinf(corrupted_embeddings).any(), "Corrupted embeddings contain Inf values!"

    
        summary_embed = global_summary_vector.expand(node_embeddings.shape[0], -1)  # [N, D]
        corrupted_summary_vector = global_summary([corrupted_embeddings])
        corrupted_summary_embed = corrupted_summary_vector.expand(corrupted_embeddings.shape[0], -1)
        loss = contrastive_loss(node_embeddings, summary_embed, corrupted_embeddings, corrupted_summary_embed)


        assert not torch.isnan(loss).any(), "Loss contains NaN values!"
        assert not torch.isinf(loss).any(), "Loss contains Inf values!"

        # Backpropagation
        loss.backward()
        # for name, param in model.named_parameters():
        #  if param.grad is not None:
        #   print(f"{name}: {param.grad.norm().item()}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
       
        optimizer.step()
        scheduler.step()  # Update learning rate
  
        # Learning Rate Monitoring
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
    
    save_dir = r"C:\Users\LENOVO\GRAPHRAG\MIE"
    os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure directory exists

    # ✅ Save embeddings only if they exist
    embeddings_path = os.path.join(save_dir, "node_embeddings.pth")
    model_path = os.path.join(save_dir, "MIEHetGRL_model.pt")

    if "node_embeddings" in locals() and node_embeddings is not None:
        torch.save(node_embeddings.detach().cpu(), embeddings_path)
        print(f"✅ Node embeddings saved at {embeddings_path}")
    else:
        print("⚠️ Warning: node_embeddings not found!")

    # ✅ Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at {model_path}")
