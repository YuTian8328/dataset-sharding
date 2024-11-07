import argparse
import torch
import webdataset as wds
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN on ImageNet using WebDataset")
    parser.add_argument('--shard_path', type=str, default="/path/to/sharded_tars/imagenet-{0000..0100}.tar", 
                        help="Path to the sharded dataset files")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--num_classes', type=int, default=1000, help="Number of output classes for the model")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs for training")
    return parser.parse_args()



# Define transformation to resize and convert PIL images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 (or any consistent size)
    transforms.ToTensor(),          # Convert PIL image to tensor
])

def create_webdataset_loader(shard_path, batch_size=32, num_workers=4):
    dataset = (
        wds.WebDataset(shard_path)
        .shuffle(1000)  # Shuffle buffer
        .decode("pil")  # Decode images as PIL images
        .to_tuple("png", "cls")  # Map .png and .cls files to (image, label)
        .map_tuple(lambda x: transform(x), int)  # Apply transformation to image, label to int
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)  # Adjusted to 32 * 56 * 56
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Output shape: (batch_size, 16, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))  # Output shape: (batch_size, 32, 56, 56)
        x = x.view(-1, 32 * 56 * 56)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    args = parse_args()
    
    # Initialize DataLoader
    dataloader = create_webdataset_loader(args.shard_path, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN(num_classes=args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print('model created')
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            print(images.shape)
            print(labels)
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            print(loss)
            optimizer.step()  # Optimization step
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print("Finished Training")

if __name__ == "__main__":
    main()

