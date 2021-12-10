import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, data, answer, numofneurons=128, epochs=10000):

        self.neuronet = nn.Sequential(
            nn.Linear(2, numofneurons),
            nn.LeakyReLU(),
            nn.Linear(numofneurons, 1),
            nn.Sigmoid()
        )

        data = np.array(data)
        self.data_range = [[np.min(data[:, 0]), np.max(data[:, 0])], [np.min(data[:, 1]), np.max(data[:, 1])]]
        data = torch.tensor(self.data_normalization(data)).float()
        answer = torch.tensor(np.array(answer)).float()
        self.training(data, answer, epochs)

    def data_normalization(self, data):
        data = np.array(data)
        out = []
        if data.ndim == 2:
            for d in data:
                out.append([(d[0] - self.data_range[0][0]) / (self.data_range[0][1] - self.data_range[0][0]),
                            (d[1] - self.data_range[1][0]) / (self.data_range[1][1] - self.data_range[1][0])])
            return np.array(out)
        else:
            d = data
            return np.array([(d[0] - self.data_range[0][0]) / (self.data_range[0][1] - self.data_range[0][0]),
                        (d[1] - self.data_range[1][0]) / (self.data_range[1][1] - self.data_range[1][0])])


    def training(self, data, answer, epochs):
        optimzer = torch.optim.SGD(self.neuronet.parameters(), lr=0.05)
        loss_func = nn.MSELoss()

        for epoch in range(epochs):
            out = self.neuronet(data)
            loss = loss_func(out, answer)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

    def check(self, features):
        features = torch.tensor(features).float()
        answer = self.neuronet(features).data
        return np.array(answer > 0.5)