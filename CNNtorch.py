import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import models
from torch import optim
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.transform import resize
import torch.nn as nn
from skimage.color import rgba2rgb
from keras.preprocessing.image import ImageDataGenerator

# Prepare data
input_dir = '/Users/akselituominen/Desktop/giftWrapML'
categories = ['books', 'bottles', 'letters', 'posters', 'shirts', 'socks', 'sportsStuff', 'tech', 'treats', 'toys']

data = []
labels = []

print('Loading images...')

# Define a transformation
transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# For each image
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        if not file.endswith('.DS_Store'):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            if img.shape[2] == 4:  # If the image has 4 channels
                img = rgba2rgb(img)  # Convert to RGB
            img = resize(img, (224, 224))
            
            # Convert NumPy array to PIL Image
            img = Image.fromarray((img * 255).astype(np.uint8))

            img = transform(img)  # Apply the transformation
            data.append(img)  # Append the processed image to data
            labels.append(category_idx)

data = torch.stack(data)
labels = torch.tensor(labels)

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels.numpy()
)

# Convert the numpy arrays back to PyTorch tensors
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)
val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=32, shuffle=False)

# Load pre-trained VGG16 model without the top layer
model = models.vgg16(pretrained=True)

# Freeze the layers of the base model

for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 1000),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1000, len(categories)),
)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model on the new data for a few epochs
for epoch in range(30):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Test performance
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

    print(f'Epoch {epoch + 1}/{30}, Loss: {val_loss / len(val_loader)}, Accuracy: {val_acc.item() / len(val_data)}')

# Save the model
torch.save(model.state_dict(), './model.pth')
