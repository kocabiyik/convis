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

    
    def forward(self, x):
        # Only store features in eval mode
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.tanh3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.tanh4(x)
        
        x = self.fc2(x)
        
        return x

    def get_feature_maps(self, x):
        """Returns stored feature maps for a given input"""
        feature_maps = {}
        x = self.conv1(x)
        feature_maps['conv1'] = x.detach().clone()
        x = self.tanh1(x)
        x = self.pool1(x)
        feature_maps['pool1'] = x.detach().clone()

        x = self.conv2(x)
        feature_maps['conv2'] = x.detach().clone()
        x = self.tanh2(x)
        x = self.pool2(x)
        feature_maps['pool2'] = x.detach().clone()

        x = self.conv3(x)
        feature_maps['conv3'] = x.detach().clone()
        x = self.tanh3(x)
        
        x = torch.flatten(x, 1)
        feature_maps['flatten'] = x.detach().clone()

        x = self.fc1(x)
        feature_maps['fc1'] = x.detach().clone()
        x = self.tanh4(x)

        x = self.fc2(x)
        feature_maps['fc2'] = x.detach().clone()

        return feature_maps
        


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
  

if __name__ == '__main__':
    main()