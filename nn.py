import pickle
import math

import numpy as np
import torch
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 606
hidden_size = 200
num_epochs = 20
learning_rate = 0.001
batch_size = 50
step_size = 3

# Fully connected neural network with one hidden layer


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


model = NeuralNet(input_size, hidden_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = list(range(0, step_size))
# val_dataset = list(range(0, 26))

print('start training')
# Train the model
for epoch in range(num_epochs):
    for i in train_dataset:
        # Move tensors to the configured device
        filename = 'models/preprocessed_{}.pickle'.format(i)
        with open(filename, 'rb') as f:
            print('loading data')
            train_data = pickle.load(f).astype(np.float32)
            print('finish_loading')

        data_size = train_data.shape[0]
        iter_count = math.ceil(data_size / batch_size)

        for k in range(iter_count):
            start = k * batch_size
            end = (k + 1) * batch_size
            if end >= data_size:
                end = data_size
            X = torch.from_numpy(train_data[start:end, 1:]).to(device)
            y = torch.from_numpy(train_data[start:end, :1]).to(device)

            # Forward pass
            Y = model(X)
            loss = criterion(Y, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Iter [{}/{}] Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, step_size, start, data_size, loss.item()), end='\r')

    print('Save model')
    torch.save(model.state_dict(), 'model.ckpt')


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: {} %'.format(
#         100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
