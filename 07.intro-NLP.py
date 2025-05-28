# In this problem, you will load in a raw body of text and set it up for training. ChatGPT uses the entire text of the internet for training, but in this problem we will use Amazon product reviews and Tweets from X.

# Your task is to encode the input dataset of strings as an integer tensor of size 
# 2
# ⋅
# N
# ×
# T
# 2⋅N×T, where 
# T
# T is the length of the longest string. The lexicographically first word should be represented as 1, the second should be 2, and so on. In the final tensor, list the positive encodings, in order, before the negative encodings.

# Inputs:

# positive - a list of strings, each with positive emotion
# negative - a list of strings, each with negative emotion


# Example:
# Input:
# positive = ["Dogecoin to the moon"]
# negative = ["I will short Tesla today"]

# Output: [
#   [1.0, 7.0, 6.0, 4.0, 0.0],
#   [2.0, 9.0, 5.0, 3.0, 8.0]
# ]

import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # Combine the positive and negative lists
        combined = positive + negative

        # Create a set of unique words and sort them
        unique_words = sorted(set(" ".join(combined).split()))

        # Create a mapping from words to indices
        word_to_index = {word: i + 1 for i, word in enumerate(unique_words)}

        # Encode all strings as integer tensors
        encoded_positive = [torch.tensor([word_to_index[word] for word in review.split()]) for review in positive]
        encoded_negative = [torch.tensor([word_to_index[word] for word in review.split()]) for review in negative]
        
        # Combine all encoded sequences but keep track of which are positive
        all_encoded = encoded_positive + encoded_negative
        
        # Pad all sequences together to the same length
        padded_sequences = torch.nn.utils.rnn.pad_sequence(all_encoded, batch_first=True, padding_value=0.0)
        
        # Split back into positive and negative
        num_positive = len(encoded_positive)
        padded_positive = padded_sequences[:num_positive]
        padded_negative = padded_sequences[num_positive:]
        
        # Concatenate in the required order (positives first, then negatives)
        dataset = torch.cat((padded_positive, padded_negative), dim=0)

        return dataset.float()

s = Solution()
positive = ["Dogecoin to the moon"]
negative = ["I will short Tesla today"]
dataset = s.get_dataset(positive, negative)
print(dataset)
