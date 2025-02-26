import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

class SimpleNeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize a simple neural network with the given layer sizes.
        
        Args:
            layer_sizes: List of integers where each integer represents the number of neurons in a layer.
                         First element is the input size, last element is the output size.
            learning_rate: The learning rate for gradient descent.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases randomly
        for i in range(1, self.num_layers):
            # Initialize weights with Xavier/Glorot initialization
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0 / layer_sizes[i-1]))
            self.biases.append(np.random.randn(layer_sizes[i], 1))
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def forward_pass(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform a forward pass through the network.
        
        Args:
            x: Input data with shape (input_size, 1)
            
        Returns:
            A tuple of (activations, zs) where:
            - activations is a list of activations for each layer
            - zs is a list of weighted inputs for each layer
        """
        activation = x
        activations = [x]  # List to store all activations
        zs = []  # List to store all weighted inputs
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
            
        return activations, zs
    
    def backward_pass(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation to compute gradients.
        
        Args:
            x: Input data with shape (input_size, 1)
            y: Target output with shape (output_size, 1)
            
        Returns:
            A tuple of (nabla_w, nabla_b) where:
            - nabla_w is a list of weight gradients
            - nabla_b is a list of bias gradients
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Forward pass
        activations, zs = self.forward_pass(x)
        
        # Backward pass
        # Calculate output error (delta)
        delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        # Calculate error for each layer
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_derivative(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            
        return nabla_w, nabla_b
    
    def update_mini_batch(self, mini_batch: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Update the network weights and biases using mini-batch stochastic gradient descent.
        
        Args:
            mini_batch: List of (x, y) tuples representing training examples
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backward_pass(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        # Update weights and biases
        self.weights = [w - (self.learning_rate / len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / len(mini_batch)) * nb 
                       for b, nb in zip(self.biases, nabla_b)]
    
    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray]], 
             epochs: int, mini_batch_size: int, 
             test_data: List[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        
        Args:
            training_data: List of (x, y) tuples representing training examples
            epochs: Number of training epochs
            mini_batch_size: Size of mini batches
            test_data: Optional list of (x, y) tuples for evaluation
        """
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            # Shuffle training data for each epoch
            np.random.shuffle(training_data)
            
            # Create mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Process each mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            
            # Evaluate on test data if provided
            if test_data:
                correct = self.evaluate(test_data)
                print(f"Epoch {j+1}: {correct} / {n_test} correct")
            else:
                print(f"Epoch {j+1} complete")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make a prediction for input x.
        
        Args:
            x: Input data with shape (input_size, 1)
            
        Returns:
            The network's prediction
        """
        activations, _ = self.forward_pass(x)
        return activations[-1]
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]) -> int:
        """
        Evaluate the network on test data.
        
        Args:
            test_data: List of (x, y) tuples representing test examples
            
        Returns:
            The number of correctly classified examples
        """
        results = [(np.argmax(self.predict(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)


# Example usage - XOR problem
if __name__ == "__main__":
    # XOR function: inputs are [0,0], [0,1], [1,0], [1,1]
    # outputs are [0], [1], [1], [0]
    X = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])
    
    Y = np.array([
        [[0]],
        [[1]],
        [[1]],
        [[0]]
    ])
    
    # Convert to format expected by neural network
    training_data = [(x.reshape(2, 1), y) for x, y in zip(X, Y)]
    
    # Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
    nn = SimpleNeuralNetwork([2, 4, 1], learning_rate=0.1)
    
    # Train the network
    print("Training neural network on XOR problem...")
    nn.train(training_data, epochs=10000, mini_batch_size=4)
    
    # Test the network
    print("\nTesting neural network:")
    for x, y in training_data:
        prediction = nn.predict(x)
        print(f"Input: {x.flatten()}, Target: {y.flatten()}, Prediction: {prediction.flatten()}")
    
    # Visualize decision boundary
    print("\nPlotting decision boundary...")
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = nn.predict(np.array([[xx[i, j]], [yy[i, j]]]))
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter([0, 0, 1, 1], [0, 1, 0, 1], c=Y.flatten(), cmap=plt.cm.RdBu)
    plt.title("Neural Network Decision Boundary for XOR Problem")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()