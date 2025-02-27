#Here we are building a neural network with 2 hidden layers and 6 neurons in each layer.
# The input layer has 4 neurons and the output layer has 2 neurons. 
# We are using the ReLU activation function for the hidden layers and the output layer has no activation function.
# The forward function is defined to pass the input through the network and return the output.
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(4, 6)
        self.second_layer = nn.Linear(6, 6)
        self.output_layer = nn.Linear(6, 2)

    def forward(self, x):
        x = torch.relu(self.first_layer(x))
        x = torch.relu(self.second_layer(x))
        output = self.output_layer(x)
        return output

model = MyModel()
print(model)