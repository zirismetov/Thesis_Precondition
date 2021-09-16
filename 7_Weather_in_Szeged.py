import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn.metrics
import torch
import warnings
import seaborn as sns
import torch.utils.data
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

data_path = '/Users/zafarzhonirismetov/Desktop/weatherHistory.csv'

data = pd.read_csv(data_path)
data.drop("Loud Cover", axis=1, inplace=True)
data['Pressure (millibars)'].replace(0, np.nan, inplace=True)
data.fillna(method='pad', inplace=True)


LEARNING_RATE = 1e-4
data_X = data.drop(['Formatted Date', 'Summary', 'Precip Type', 'Daily Summary',
                    'Apparent Temperature (C)', 'Temperature (C)'], axis=1)

class DatasetLoadBoston(torch.utils.data.Dataset):

    def __init__(self):
        self.X = data_X.to_numpy()
        np_x = np.copy(self.X)
        for i in range(np_x.shape[-1]):
            self.X[:, i] = ((np_x[:, i] - np.min(np_x[:, i])) / (np.max(np_x[:, i]) - np.min(np_x[:, i])))

        self.y = data['Temperature (C)'].to_numpy()
        np_y = np.copy(self.y)
        self.y[:] = ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])))
        self.y = np.expand_dims(self.y, axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


dataset = DatasetLoadBoston()

tr_idx = np.arange(len(dataset))

subset_train_data, subset_test_data = train_test_split(
    tr_idx,
    test_size=0.20,
    random_state=0)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=64,
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=64,
                                              shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=5, out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model()
opt = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

losses_train = []
losses_test = []
R2_train = []
R2_test = []
for epoc in range(1, 15):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        R2_s = []
        model.train()
        if dataloader is dataloader_test:
            model.eval()
        for x, y in dataloader:

            y_prim = model.forward(torch.FloatTensor(x.float()))
            y = torch.FloatTensor(y.float())

            loss = torch.mean(torch.abs(y - y_prim))
            R2 = sklearn.metrics.r2_score(y.detach(), y_prim.detach())

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
        f'R2_test {round(R2_test[-1], 4)} ')

plt.subplot(2, 1, 1)
plt.title('Loss graph')
plt.xlabel('epochs')
plt.plot(losses_train, label="loss_trian")
plt.plot(losses_test, label="loss_test")
plt.legend(loc='upper right', shadow=False, fontsize='medium')
for var in (losses_train, losses_test):
    plt.annotate('%0.2f' % min(var), xy=(1, min(var)), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.subplot(2, 1, 2)
plt.title('R2')
plt.xlabel('epochs')
plt.plot(R2_train, label="R2_trian")
plt.plot(R2_test, label="R2_test")
plt.legend(loc='lower right', shadow=False, fontsize='medium')
for var in (R2_train, R2_test):
    plt.annotate('%0.2f' % max(var), xy=(1, max(var)), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.tight_layout()
plt.show()
