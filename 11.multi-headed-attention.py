import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        head_size = attention_dim // num_heads
        
        # Instantiate num_heads SingleHeadAttention modules.
        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, head_size)
            for _ in range(num_heads)
        ])
        # Force all heads to share the same parameters.
        ref_state = self.heads[0].state_dict()
        for head in self.heads[1:]:
            head.load_state_dict(ref_state)
        
        # Final linear layer that combines heads.
        # Set to identity mapping so that output equals concatenated heads.
        self.linear = nn.Linear(attention_dim, attention_dim, bias=False)
        self.linear.weight.data = torch.eye(attention_dim)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded shape: [batch_size, context_length, embedding_dim]
        head_outputs = [head(embedded) for head in self.heads]  # Each: [B, T, head_size]
        concatenated = torch.cat(head_outputs, dim=-1)  # [B, T, num_heads*head_size] = [B, T, attention_dim]
        output = self.linear(concatenated)  # With identity mapping, output equals concatenated.
        return output
    
    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, head_dim: int):
            super().__init__()
            torch.manual_seed(0)
            # Define Key, Query, and Value generators with no bias.
            self.key_gen = nn.Linear(embedding_dim, head_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, head_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, head_dim, bias=False)
        
        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # embedded: [B, T, embedding_dim]
            k = self.key_gen(embedded)    # [B, T, head_dim]
            q = self.query_gen(embedded)    # [B, T, head_dim]
            v = self.value_gen(embedded)    # [B, T, head_dim]
            
            # Compute scores: Q x Kᵀ, scaled by sqrt(head_dim).
            scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
            
            # Create a lower-triangular mask so that each token can only attend to itself and previous tokens.
            B, T, _ = scores.shape
            mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=embedded.device))
            scores = scores.masked_fill(~mask, float('-inf'))
            
            # Compute attention weights with softmax.
            attention_weights = F.softmax(scores, dim=-1)
            # Multiply weights by v to get output.
            output = torch.matmul(attention_weights, v)  # [B, T, head_dim]
            return output


