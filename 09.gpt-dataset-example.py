import torch
from typing import List, Tuple

class Solution:
    # Return the input dataset X and the labels Y. len(X) = len(Y).
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # Split the raw text into words
        words = raw_dataset.split()
        
        # Generate batch_size different random starting indices in the appropriate range
        torch.manual_seed(0)
        # The maximum starting index is len(words) - context_length to ensure a full context and target
        indices = torch.randint(0, len(words) - context_length, (batch_size,))
        
        X = []
        Y = []
        for idx in indices:
            start = idx.item()  # Convert tensor scalar to a Python integer
            # The context consists of context_length words
            context = words[start : start + context_length]
            # The target is the sequence of the next context_length words (shifted by one)
            target = words[start + 1 : start + context_length + 1]
            X.append(context)
            Y.append(target)
        return X, Y

# Example usage:
s = Solution()
raw_dataset = "Hello darkness my old friend"
context_length = 3
batch_size = 2
X, Y = s.batch_loader(raw_dataset, context_length, batch_size)
print("X =", X)
print("Y =", Y)
