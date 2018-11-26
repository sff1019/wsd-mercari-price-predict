import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 606
hidden_size = 200


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
model.load_state_dict(torch.load('model.ckpt'))
# criterion = nn.MSELoss()


# test_dataset = [0]
test_dataset = list(range(0, 7))

with torch.no_grad():
    df = pd.DataFrame({'price': []})
    for i in test_dataset:
        with open('models/preprocessed_test_%d.pickle' % i, 'rb') as f:
            print('loading data')
            test_data = pickle.load(f).astype(np.float32)
            print('finish_loading')

        X = torch.from_numpy(test_data).to(device)
        # y = torch.from_numpy(train_data[:, :1]).to(device)
        Y = model(X)
        # loss = criterion(Y, y)
        Y = Y.cpu().data.numpy()[:, 0]
        new_df = pd.DataFrame({'price': Y})
        df = pd.concat([df, new_df], ignore_index=True)
        print(df)

    print('Start creating csv file')
    df.to_csv('test.csv')
    print('Finish')
