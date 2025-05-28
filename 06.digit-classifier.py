import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

# Define the neural network model.
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # One linear layer from 512 inputs to 10 outputs.
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        # x should be of shape [batch_size, 512]
        out = self.fc(x)
        # Apply ReLU to ensure non-negative values
        activations = F.relu(out)
        # Normalize to create proper probabilities that sum to 1
        # Add small epsilon to avoid division by zero
        sum_activations = torch.sum(activations, dim=1, keepdim=True) + 1e-8
        probabilities = activations / sum_activations
        return probabilities

# Instantiate the model.
model = DigitClassifier()

# Define image preprocessing for training (using MNIST) and for your custom digit image.
# For training on MNIST, we use MNIST's standard normalization values.
train_transform = transforms.Compose([
    transforms.Resize((16, 32)),  # Resize MNIST images (originally 28x28) to 16x32 (512 pixels)
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# For your custom image, you might stick with a simple normalization.
inference_transform = transforms.Compose([
    transforms.Grayscale(),          # Ensure image is single channel.
    transforms.Resize((16, 32)),     # Resize to 16x32.
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST training and test datasets.
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training settings.
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects raw logits.
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# Training loop.
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Flatten images from [batch_size, 1, 16, 32] to [batch_size, 512]
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(images)
        # Since outputs already go through softmax in our model,
        # you could either remove softmax in the model or use NLLLoss with log-softmax.
        # For simplicity, assume we remove softmax during training. Here's an adjustment:
        logits = model.fc(images)  # Get raw logits.
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate on test set.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy: {:.2f}%".format(100 * correct / total))

# Now, perform inference on your custom image 'digit.png'
image_path = "digit.png"
image = Image.open(image_path)
image_tensor = inference_transform(image)
# Flatten the image to [1, 512]
image_tensor = image_tensor.view(1, -1)

with torch.no_grad():
    output_probabilities = model(image_tensor)
    predicted_class = torch.argmax(output_probabilities, dim=1).item()

# Round probabilities to 4 decimal places.
rounded_probs = torch.round(output_probabilities * 1e4) / 1e4

print("Predicted digit:", predicted_class)
print("Output probabilities:", rounded_probs.tolist())
print("Most likely digit:", torch.argmax(output_probabilities).item())