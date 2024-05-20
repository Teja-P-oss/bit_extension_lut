import torch
import torch.nn as nn
import torch.nn.functional as F

class LSBPredictionCNN(nn.Module):
    def __init__(self):
        super(LSBPredictionCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming input image size of 64x64
        self.fc2 = nn.Linear(128, 4)  # Output layer with 4 classes (2 bits extension)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the output for fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = LSBPredictionCNN()
# Print the model architecture
print(model)






import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, low_bit_images, high_bit_images, transform=None):
        self.low_bit_images = low_bit_images
        self.high_bit_images = high_bit_images
        self.transform = transform

    def __len__(self):
        return len(self.low_bit_images)

    def __getitem__(self, idx):
        low_bit_img = self.low_bit_images[idx]
        high_bit_img = self.high_bit_images[idx]
        
        if self.transform:
            low_bit_img = self.transform(low_bit_img)
            high_bit_img = self.transform(high_bit_img)
        
        return low_bit_img, high_bit_img

# Define your low and high bit depth image data
low_bit_images = ...
high_bit_images = ...

# Define transformations (you may need to adjust this based on your data)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    # Add more transformations if needed (e.g., normalization)
])

# Create a dataset instance
dataset = CustomDataset(low_bit_images, high_bit_images, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader instances for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the CNN model
class LSBPredictionCNN(nn.Module):
    # Define model architecture ...

# Instantiate the model
model = LSBPredictionCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Validation loop
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss = val_loss / len(val_dataset)
val_accuracy = 100 * correct / total
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")















import torch
import itertools
import numpy as np

# Assume the input images are grayscale with an original bit depth of 8
# and we're extending the bit depth by 2 bits, resulting in 4 classes (00, 01, 10, 11)
bit_depth = 8
num_bits_to_extend = 2
num_classes = 2 ** num_bits_to_extend

# Generate all possible input configurations
def generate_input_configs(bit_depth, context_size=1):
    possible_values = range(2 ** bit_depth)
    configs = list(itertools.product(possible_values, repeat=context_size))
    return configs

# For simplicity, assume context size is 1 (current pixel value only)
context_size = 1
input_configs = generate_input_configs(bit_depth, context_size)

# Load the trained model (assuming model is already trained and saved)
model = LSBPredictionCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Initialize the lookup table
lut = {}

# Predict LSB fillings for each input configuration
with torch.no_grad():
    for config in input_configs:
        # Convert the input configuration to a tensor
        input_tensor = torch.tensor(config, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Predict the output using the trained model
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        
        # Store the prediction in the lookup table
        lut[config] = predicted_class.item()

# Save the LUT to a file for later use
np.save('lut.npy', lut)

# To load the LUT later, use:
# lut = np.load('lut.npy', allow_pickle=True).item()






