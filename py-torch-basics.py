import torch
from torchtyping import TensorType
import torch.nn.functional

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        # reshape the tensor to (M*N//2) X 2 in memory
        M, N = to_reshape.shape
        reshaped = torch.reshape(to_reshape, (M*N//2, 2))
        return torch.round(reshaped, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        avg = torch.mean(to_avg, dim=0)  # Average along dimension 0 (rows)
        return torch.round(avg, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        result = torch.cat((cat_one, cat_two), 1)  # Concatenate along dimension 1 (columns)
        return torch.round(result, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Use torch.nn.functional.mse_loss for MSE calculation
        mse_loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(mse_loss, decimals=4)

def main():
    # Test reshape function
    to_reshape = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ])
    
    # Test average function
    to_avg = torch.tensor([
        [0.8088, 1.2614, -1.4371],
        [-0.0056, -0.2050, -0.7201]
    ])
    
    # Test concatenate function
    cat_one = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    cat_two = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    
    # Test get_loss function
    prediction = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    
    solution = Solution()
    
    print("Reshape Test:")
    print(solution.reshape(to_reshape))
    
    print("\nAverage Test:")
    print(solution.average(to_avg))
    
    print("\nConcatenate Test:")
    print(solution.concatenate(cat_one, cat_two))
    
    print("\nGet Loss Test:")
    print(solution.get_loss(prediction, target))

if __name__ == "__main__":
    main()