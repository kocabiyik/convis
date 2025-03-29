import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Layer definitions
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        
        self.fc1 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, num_classes)
        
        # Feature maps storage (only used in eval mode)
        self._feature_maps = {}
    
    def forward(self, x):
        # Only store features in eval mode
        store_features = not self.training
        
        if store_features:
            self._feature_maps.clear()
            self._feature_maps['input'] = x.detach().clone()
        
        x = self.conv1(x)
        if store_features: self._feature_maps['conv1'] = x.detach().clone()
        x = self.tanh1(x)
        x = self.pool1(x)
        if store_features: self._feature_maps['pool1'] = x.detach().clone()
        
        x = self.conv2(x)
        if store_features: self._feature_maps['conv2'] = x.detach().clone()
        x = self.tanh2(x)
        x = self.pool2(x)
        if store_features: self._feature_maps['pool2'] = x.detach().clone()
        
        x = self.conv3(x)
        if store_features: self._feature_maps['conv3'] = x.detach().clone()
        x = self.tanh3(x)
        
        x = torch.flatten(x, 1)
        if store_features: self._feature_maps['flatten'] = x.detach().clone()
        
        x = self.fc1(x)
        if store_features: self._feature_maps['fc1'] = x.detach().clone()
        x = self.tanh4(x)
        
        x = self.fc2(x)
        if store_features: self._feature_maps['output'] = x.detach().clone()
        
        return x
    
    def get_feature_maps(self):
        """Returns stored feature maps if in eval mode, None otherwise"""
        if self.training:
            return None
        return self._feature_maps
    
    def clear_feature_maps(self):
        """Clears the stored feature maps"""
        self._feature_maps.clear()

def visualize_feature_maps(model, test_loader, device, num_samples=1):
    """Helper function to visualize feature maps"""
    model.eval()  # Ensure we're in eval mode
    
    # Get one batch of data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    
    # Forward pass to capture features
    with torch.no_grad():
        _ = model(images[:num_samples])  # Process only a few samples
    
    # Retrieve and return feature maps
    feature_maps = model.get_feature_maps()
    model.clear_feature_maps()
    
    return feature_maps, images[:num_samples], labels[:num_samples]


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({'Loss': loss.item(), 'Acc': 100.*correct/total})
    
    train_loss /= len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Evaluation function
def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transform, 
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=False, 
                                           transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'lenet5_best.pth')
    
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    
    # Example visualization usage
    print("\nCapturing feature maps for visualization...")
    feature_maps, sample_images, sample_labels = visualize_feature_maps(
        model, test_loader, device, num_samples=3
    )
    
    # feature_maps now contains all intermediate representations
    # sample_images and sample_labels contain the corresponding inputs
    print("Available feature maps:", list(feature_maps.keys()))
    print("Sample images shape:", sample_images.shape)
    print("Sample labels:", sample_labels)

if __name__ == '__main__':
    main()