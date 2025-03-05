import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mnist_loader import load_mnist
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and preprocess MNIST data
(train_images, train_labels), (test_images, test_labels) = load_mnist()

# Convert numpy arrays to PyTorch tensors
train_images = torch.FloatTensor(train_images.reshape(-1, 784))
train_labels = torch.LongTensor(train_labels)
test_images = torch.FloatTensor(test_images.reshape(-1, 784))
test_labels = torch.LongTensor(test_labels)

# Create data loaders
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss function, and optimizer
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Evaluation function
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    # Train the model
    print('Starting training...')
    train(epochs=10)
    
    # Evaluate on test set
    print('\nEvaluating model...')
    evaluate()