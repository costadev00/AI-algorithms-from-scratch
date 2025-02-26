import numpy as np
from numpy.typing import NDArray


# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:
    # X is an Nx3 NumPy array
    # weights is a 3x1 NumPy array
    # HINT: np.matmul() will be useful
    # return np.round(your_answer, 5)
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = np.matmul(X, weights)
        return np.round(prediction, 5)

    # model_prediction is an Nx1 NumPy array
    # ground_truth is an Nx1 NumPy array
    # HINT: np.mean(), np.square() will be useful
    # return round(your_answer, 5)
    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        #implement mean squared error
        #formula: 1/N * sum((model_prediction - ground) ^ 2)
        error = np.mean(np.square(model_prediction - ground_truth))
        return round(error, 5)
