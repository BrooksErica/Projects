import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

#define CNN model 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 for MNIST
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#my images
class CharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        
        for label in self.class_map:
            path = os.path.join(root_dir, label)
            for img_name in os.listdir(path):
                self.images.append(os.path.join(path, img_name))
                self.labels.append(self.class_map[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

#function to show images
def show_images(dataloader, num_images=5, title="Sample Images"):
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Label: {label}')
    plt.suptitle(title)
    plt.show()

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#prepare MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#show some MNIST images
print("Displaying sample MNIST images:")
show_images(train_loader, title="MNIST Training Images")

#initialize model, loss function, and optimizer
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train on MNIST
print("Training on MNIST...")
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("MNIST training completed!")

#modify the last layer for 5 classes (A-E)
model.fc2 = nn.Linear(128, 5).to(device)

#prepare your character dataset
char_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # MNIST size
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

char_dataset = CharacterDataset(root_dir='./letters', transform=char_transform)
char_loader = DataLoader(char_dataset, batch_size=32, shuffle=True)

#show some character images
print("\nDisplaying sample character images:")
show_images(char_loader, title="Custom Character Dataset Images")

#fine-tune on character dataset
print("Fine-tuning on character dataset...")
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning

num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(char_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(char_loader)}], Loss: {loss.item():.4f}')

print("Transfer learning completed!")

#function to show predictions
def show_predictions(model, dataloader, num_images=5):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    
    for i, (img, label, pred) in enumerate(zip(images, labels, predicted)):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
        true_label = label_map[label.item()]
        pred_label = label_map[pred.item()]
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}')
    
    plt.suptitle("Model Predictions on Character Dataset")
    plt.show()

#show some predictions
print("\nDisplaying model predictions:")
show_predictions(model, char_loader)

#save the model
torch.save(model.state_dict(), './letters_model.pth')