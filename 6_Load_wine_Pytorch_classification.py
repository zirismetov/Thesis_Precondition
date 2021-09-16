import torch
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Process hyper-parameters')

parser.add_argument('-lr', type=float, default= 1e-4)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-test_size', type=float, default=0.20)

args = parser.parse_args()

print(args.lr, args.batch_size, args.test_size)
# LEARNING_RATE = 1e-4
# BATCH_SIZE = 32
# DEVICE = 'cpu'
class DatasetLoadWines(torch.utils.data.Dataset):
    def __init__(self):
        dataset = sklearn.datasets.load_wine()
        self.X = dataset.data
        self.Y = dataset.target

        np_x = np.array(self.X)
        for i in range(np_x.shape[-1]):
            self.X[:, i] = (np_x[:, i] - np.min(np_x[:, i])) / (np.max(np_x[:, i]) - np.min(np_x[:, i]))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


data = DatasetLoadWines()
tr_inx = np.arange(len(data))
subset_train_data, subset_test_data = train_test_split(
    tr_inx,
    test_size=args.test_size,
    random_state=0,
    stratify=data.Y)

dataset_train = torch.utils.data.Subset(data, subset_train_data)
dataset_test = torch.utils.data.Subset(data, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=args.batch_size,
                                              shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=13, out_features=39),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=39, out_features=39),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=39, out_features=26),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=26, out_features=3),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)

loss_train = []
loss_test = []
acc_train = []
acc_test = []
f1_train = []
f1_test = []


def compute_accuracy(y_prim, y):
    num_sample = y_prim.shape[0]
    pred_idx = np.argmax(y_prim, axis=1)
    true_idx = np.argmax(y, axis=1)
    acc = np.sum(np.equal(pred_idx, true_idx).astype(float)) / num_sample

    return acc


def f1_score(y, y_prim):
    pred_idx = np.argmax(y_prim, axis=1)
    true_idx = np.argmax(y, axis=1)
    conf_matrix = np.zeros((y.shape[-1], y.shape[-1]))

    skf1 = sklearn.metrics.f1_score(true_idx, pred_idx, average='macro')
    for i in range(len(pred_idx)):
        conf_matrix[pred_idx[i], true_idx[i]] += 1
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greens'))
    plt.xticks([0, 1, 2], ['Red', 'White', 'Rose'])
    for x in range(3):
        for y in range(3):
            plt.annotate(
                str(round(100 * conf_matrix[x,y]/ ((np.sum(conf_matrix[x])+ 1e-8)), 1)), xy= (y, x),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor= 'white'
            )
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig('conf_matrix')
    plt.close()
    # f1 = (2 * conf_matrix[0, 0] / (2 * (conf_matrix[0, 0]
    #                                     + (conf_matrix[0, 1] + conf_matrix[0, 2])
    #                                     + (conf_matrix[1, 0] + conf_matrix[2, 0])))).astype(float)
    # if skf1 == f1:
    #     print('Skf1 and f1_mine is equal')
    # else:
    #     print('Skf1 and f1_mine is FALSE')


    return skf1


for epoch in range(1, 30):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accuracies = []
        f1s = []
        for x, y in dataloader:

            y_prim = model.forward(x.float())

            np_y = np.array(y)
            y = np.zeros((len(np_y), 3))
            y[np.arange(len(np_y)), np_y] = 1.0

            f1 = f1_score(y, y_prim.data.numpy())
            loss = torch.mean(-torch.FloatTensor(y) * torch.log(y_prim + 1e-8))
            acc = compute_accuracy(y_prim.data.numpy(), y)

            losses.append(loss.item())
            accuracies.append(acc)
            f1s.append(f1)
            if dataloader is dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        if dataloader is dataloader_train:
            loss_train.append(np.mean(losses))
            acc_train.append(np.mean(accuracies))
            f1_train.append(np.mean(f1s))

        else:
            loss_test.append(np.mean(losses))
            acc_test.append(np.mean(accuracies))
            f1_test.append(np.mean(f1s))

    print(
        f'epoch: {epoch} '
        f'loss_train: {round(loss_train[-1], 3)} '
        f'loss_test: {round(loss_test[-1], 3)} '
        f'acc_train: {round(acc_train[-1], 3)} '
        f'acc_test: {round(acc_test[-1], 3)} '
        f'f1_train: {round(f1_train[-1], 3)} '
        f'f1_test: {round(f1_test[-1], 3)} '
    )
plt.subplot(3, 1, 1)
plt.title('loss')
plt.plot(loss_train, label="loss_trian")
plt.plot(loss_test, label="loss_test")
plt.legend(loc='upper right', shadow=False, fontsize='medium')

plt.subplot(3, 1, 2)
plt.title('acc')
plt.plot(acc_train, label="acc_trian")
plt.plot(acc_test, label="acc_trian")
plt.legend(loc='lower right', shadow=False, fontsize='medium')

plt.subplot(3, 1, 3)
plt.title('f1_s')
plt.plot(f1_train, label="f1_trian")
plt.plot(f1_test, label="f1_test")
plt.legend(loc='lower right', shadow=False, fontsize='medium')

plt.tight_layout()
plt.show()
