# Importieren der benötigten Bibliotheken
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Hyperparameter
batch_size = 64
learning_rate = 0.001
num_epochs = 5
# MNIST-Datensatz herunterladen und transformieren (Normierung der Bilder)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisieren der Graustufenbilder
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Definition des CNN-Modells
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Erstes Convolutional Layer, gefolgt von MaxPooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Zweites Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Fully connected Layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 32 Channels und 7x7 Feature Map nach Pooling
        self.fc2 = nn.Linear(128, 10)  # 10 Klassen für die Ziffern 0–9
    def forward(self, x):
        # Forward-Pass durch das Netzwerk
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flachlegen des Tensors
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Initialisierung des Modells, Verlustfunktion und Optimierer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training des Modells
def train_model():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Nullsetzen der Gradienten
            optimizer.zero_grad()
            # Vorwärtsdurchlauf
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Rückwärtsdurchlauf und Optimierung
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
# Testen des Modells
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
# Ausführen des Trainings und Testens
train_model()
test_model()
# Speichern des trainierten Modells
torch.save(model.state_dict(), 'cnn_mnist.pth')