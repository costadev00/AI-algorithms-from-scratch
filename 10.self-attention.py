import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Instantiate linear layers in order: Key, Query, Value with no bias.
        self.linear_key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.linear_query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.linear_value = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded is of shape [B, T, embedding_dim]
        # Generate Key, Query, Value matrices.
        K = self.linear_key(embedded)   # shape: [B, T, attention_dim]
        Q = self.linear_query(embedded)   # shape: [B, T, attention_dim]
        V = self.linear_value(embedded)   # shape: [B, T, attention_dim]
        
        # Compute attention scores: Dot product between Q and Kᵀ gives [B, T, T].
        # Scale by sqrt(attention_dim) for numerical stability.
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
        
        # Create a lower-triangular mask for the sequence.
        # We want to mask out future tokens (upper triangle entries).
        B, T, _ = attention_scores.shape
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=embedded.device))
        # Set positions where mask is False to -inf so that softmax will turn them to 0.
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax along the sequence dimension (dim=-1) to get attention weights.
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Multiply attention weights with V to produce the output.
        output = torch.matmul(attention_weights, V)  # Shape: [B, T, attention_dim]
        
        # Round the output to 4 decimal places.
        output = torch.round(output * 10000) / 10000.0
        
        return output
