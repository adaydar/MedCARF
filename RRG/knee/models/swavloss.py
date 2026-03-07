import torch
import torch.nn as nn
import torch.nn.functional as F

class SwAV(nn.Module):
    def __init__(self, feature_dim=768, num_prototypes=300):
        super().__init__()
        self.prototypes = nn.Linear(feature_dim, num_prototypes, bias=False)

    def forward(self, features):
        return self.prototypes(features)  # Get soft cluster assignments

def sinkhorn_knopp(Q, eps=0.05, iters=3):
    """ Sinkhorn-Knopp optimization for balanced cluster assignments """
    Q = torch.exp(Q / eps)  # Softmax normalization
    Q /= Q.sum(dim=1, keepdim=True)  # Normalize rows

    for _ in range(iters):
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize columns
        Q /= Q.sum(dim=1, keepdim=True)  # Normalize rows
    
    return Q.detach()

def swav_loss(features_I, features_I_prime):
    """
    Compute SwAV Swapped Prediction Loss:
    - Assigns cluster prototypes to both I and I'
    - Uses Sinkhorn-Knopp to balance assignments
    - Computes cross-entropy loss between swapped predictions
    """
    swav = SwAV()
    device = features_I.device
    features_I_prime = features_I_prime.to(device)
    swav = swav.to(device)
    # Compute cluster assignments for both
    assignments_I = swav(features_I)       # Cluster assignments for I
    assignments_I_prime = swav(features_I_prime)  # Cluster assignments for I'

    # Balance assignments using Sinkhorn-Knopp
    assignments_I = sinkhorn_knopp(assignments_I)
    assignments_I_prime = sinkhorn_knopp(assignments_I_prime)

    # Swapped prediction loss: Predict I' from I and vice versa
    loss_I = -torch.mean(torch.sum(assignments_I * torch.log(assignments_I_prime + 1e-10), dim=1))
    loss_I_prime = -torch.mean(torch.sum(assignments_I_prime * torch.log(assignments_I + 1e-10), dim=1))
    
    return (loss_I + loss_I_prime) / 2  # Average loss

# batch_size = 8
# feature_dim = 768
# logit_dim = 14

# # Generate random tensors for features and logits
# features_I = torch.randn(batch_size, feature_dim)
# features_I_prime = torch.randn(batch_size, feature_dim)
# logits_I = torch.randn(batch_size, logit_dim)
# logits_I_prime = torch.randn(batch_size, logit_dim)
# loss = swav_loss(features_I, features_I_prime,logits_I,logits_I_prime)
# print(loss)