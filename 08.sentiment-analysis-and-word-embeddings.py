import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Define the embedding layer
        self.embedding = nn.Embedding(vocabulary_size, 16)  # Embedding dimension = 16
        # Define the linear layer
        self.linear = nn.Linear(16, 1)  # Maps the averaged embedding to a single output

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Pass input through the embedding layer
        x = self.embedding(x)  # Shape: [B, T, embed_dim]
        # Average over the sequence length (dim=1)
        x = torch.mean(x, dim=1)  # Shape: [B, embed_dim]
        # Pass through the linear layer
        x = self.linear(x)  # Shape: [B, 1]
        # Apply sigmoid activation to get values between 0 and 1
        x = torch.sigmoid(x)
        # Round to 2 decimal places
        x = torch.round(x * 100) / 100.0
        return x

# Instantiate the model
vocabulary_size = 170000
model = Solution(vocabulary_size)

# Example input
x = torch.tensor([
    [2, 7, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0],  # "The movie was okay"
    [1, 4, 12, 3, 10, 5, 15, 11, 6, 9, 13, 7]  # "I don't think anyone should ever waste their money on this movie"
], dtype=torch.long)

# Forward pass
output = model(x)
print(output)