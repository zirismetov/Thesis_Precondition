import sklearn.datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.model_selection import train_test_split

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
class DatasetLoadBoston(torch.utils.data.Dataset):

    def __init__(self):
        dataset = sklearn.datasets.load_boston()
        self.X = dataset.data
        self.y = dataset.target
        np_x = np.copy(self.X)
        for i in range(np_x.shape[-1]):
            self.X[:, i] = ((np_x[:, i] - np.min(np_x[:, i])) / (np.max(np_x[:, i]) - np.min(np_x[:, i])))

        np_y = np.copy(self.y)
        self.y[:] = ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])))
        self.y = np.expand_dims(self.y, axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


data = DatasetLoadBoston()

tr_indx = np.arange(len(data))

subset_train_data, subset_test_data = train_test_split(
    tr_indx,
    test_size=0.33,
    random_state=0)

dataset_train = torch.utils.data.Subset(data, subset_train_data)
dataset_test = torch.utils.data.Subset(data, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=13, out_features=26),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=26, out_features=1),
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model()
opt = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def R2_compute(y, y_prim):
    sum_of_errors = torch.sum((y - y_prim) ** 2)
    denominator = torch.sum((y - torch.mean(y)) ** 2)
    return 1 - sum_of_errors / denominator


losses_train = []
losses_test = []
R2_train = []
R2_test = []
for epoc in range(1, 150):

    for dataloader in [dataloader_train, dataloader_test]:

        losses = []
        R2_s = []

        for x, y in dataloader:

            y_prim = model.forward(torch.FloatTensor(x.float()))
            y = torch.FloatTensor(y.float())

            loss = torch.mean(torch.abs(y - y_prim))
            R2 = R2_compute(y, y_prim)

            losses.append(loss.item())
            R2_s.append(R2.item())
            if dataloader is dataloader_train:
                loss.backward()
                opt.step()
                opt.zero_grad()

        if dataloader is dataloader_train:
            losses_train.append(np.mean(losses))
            R2_train.append(np.mean(R2_s))

        else:
            losses_test.append(np.mean(losses))
            R2_test.append(np.mean(R2_s))

    print(
        f'epoch: {epoc} '
        f'losses_train: {round(losses_train[-1], 4)} '
        f'losses_test: {round(losses_test[-1], 4)} '
        f'R2_train: {round(R2_train[-1], 4)} '
        f'R2_test {round(R2_test[-1], 4)}')

plt.subplot(2, 1, 1)
plt.title('loss')
plt.plot(losses_train, label="loss_trian")
plt.plot(losses_test, label="loss_test")
plt.legend(loc='upper right', shadow=False, fontsize='medium')

plt.subplot(2, 1, 2)
plt.title('R2')
plt.plot(R2_train, label="R2_trian")
plt.plot(R2_test, label="R2_test")
plt.legend(loc='lower right', shadow=False, fontsize='medium')
plt.show()