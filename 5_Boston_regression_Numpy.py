import sklearn.datasets
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data

import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
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
    test_size=0.20,
    random_state=0)

dataset_train = torch.utils.data.Subset(data, subset_train_data)
dataset_test = torch.utils.data.Subset(data, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)


class Variable:
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LayerLinear():
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.random((out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(

            np.matmul(self.W.value, self.x.value.transpose()).transpose() + self.b.value
        )
        return self.output  # this output will be input for next function in model

    def backward(self):

        self.b.grad = 1 * self.output.grad


        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )

        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class LeakyReLU:
    def __init__(self):
        self.x = None
        self.output: Variable = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        z = x
        for i in range(len(x.value)):
            for j in range(len(x.value[0])):
                if z.value[i, j] < 0:
                    z.value[i, j] *= 0.01

        self.output = z
        return self.output

    def backward(self):
        self.x.grad = np.ones_like(self.x.value)
        self.x.grad[self.x.value < 0] = 0.01


class LossMAE:
    def __init__(self):
        self.y: Variable = None
        self.y_prim: Variable = None
        self.rmse: Variable = None

    def forward(self, y: Variable, y_prim: Variable) -> float:
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad = - (self.y.value - self.y_prim.value) / np.abs(self.y.value - self.y_prim.value)


class Dropout:
    def __init__(self, keep_rate):
        self.keep_rate = keep_rate
        self.drop = []
        self.x = None
        self.output: Variable = None

    def forward(self, x: Variable, is_train):
        if is_train:
            self.x = x
            unit = x
            drop_chane = np.random.rand(unit.value.shape[0], unit.value.shape[1])
            self.drop = drop_chane < self.keep_rate
            unit.value = (unit.value * self.drop) / self.keep_rate
            self.output = unit
            return self.output

    def backward(self):
        self.x.grad = (self.x.grad * self.drop) / self.keep_rate


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=13, out_features=26),
            LeakyReLU(),
            # Dropout(keep_rate=0.8),
            # LayerLinear(in_features=56, out_features=26),
            # LeakyReLU(),
            LayerLinear(in_features=26, out_features=1)
        ]

    def forward(self, x: Variable, is_train):
        out = x

        for layer in self.layers:
            if is_train:
                if isinstance(layer, Dropout):
                    out = layer.forward(out, is_train)
                else:
                    out = layer.forward(out)
            else:
                if isinstance(layer, Dropout):
                    continue
                out = layer.forward(out)

        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= self.learning_rate * np.mean(param.grad, axis=0)



def R2_compute(y, y_prim):
    sum_of_errors = np.sum((y.value - y_prim.value) ** 2)
    denominator = np.sum((y.value - np.mean(y.value)) ** 2)
    k = sklearn.metrics.r2_score(y.value, y_prim.value)
    r2 = 1 - sum_of_errors / denominator
    return r2

model = Model()
opt = OptimizerSGD(
    parameters=model.parameters(),
    learning_rate=LEARNING_RATE)
loss_fn = LossMAE()

losses_train = []
losses_test = []
R2_train = []
R2_test = []
for epoch in range(1, 100):

    is_train = True
    for dataset in [dataloader_train, dataloader_test]:

        losses = []
        R2_s = []
        if dataset is dataloader_test:
            is_train = False

        for x, y in dataset:

            x = x.numpy()
            y = y.numpy()

            y_prim = model.forward(Variable(x), is_train=is_train)

            loss = loss_fn.forward(Variable(y), y_prim)

            R2 = R2_compute(Variable(y), y_prim)

            R2_s.append(R2)
            losses.append(loss)

            if dataset is dataloader_train:
                loss_fn.backward()
                model.backward()
                opt.step()



        if dataset is dataloader_train:
            losses_train.append(np.mean(losses))
            R2_train.append(np.mean(R2_s))
        else:
            losses_test.append(np.mean(losses))
            R2_test.append(np.mean(R2_s))

    print(f'epoch: {epoch} '
          f'losses_train: {round(losses_train[-1], 3)} '
          f'losses_test: {round(losses_test[-1], 3)} '
          f'R2_train: {round(R2_train[-1], 3)} '
          f'R2_test: {round(R2_test[-1], 3)}')

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



#