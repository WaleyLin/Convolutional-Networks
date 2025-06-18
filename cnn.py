# import torch and other necessary modules from torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((100, 100)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Fix normalization for RGB images
])

dataset = datasets.ImageFolder(root='./petimages', transform=transform)

test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# model hyperparameters
learning_rate = 0.0001
batch_size = 32
epoch_size = 10

# model design 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'  
cnn = CNN().to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

# start model training
cnn.train()
for epoch in range(epoch_size):  
    loss = 0.0  

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss_value = criterion(outputs, labels)
        loss_value.backward()
        optimizer.step()

        loss += loss_value.item()  
        if i % 100 == 99:  
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')

# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval()  
with torch.no_grad():  
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        ground_truth += labels.cpu().tolist()  
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        prediction += predicted.cpu().tolist()  

accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='macro', zero_division=1)
precision = precision_score(ground_truth, prediction, average='macro', zero_division=1)
