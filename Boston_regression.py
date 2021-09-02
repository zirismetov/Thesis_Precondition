# # Boston NUMPY regression
# import time
# import sys
# import sklearn.datasets
# import numpy as np
# import matplotlib.pyplot as plt
#
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 11
#
# X, Y = sklearn.datasets.load_boston(return_X_y=True)
#
# np_y = np.copy(Y)
# Y[:] = 2 * ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])) - 0.5)
# Y = np.expand_dims(Y, axis=1)
#
# np_x = np.copy(X)
# for x in range(13):
#     X[:, x] = 2 * ((np_x[:, x] - np.min(np_x[:, x])) / (np.max(np_x[:, x]) - np.min(np_x[:, x])) - 0.5)
#
# np.random.seed(0)
# idxes_rand = np.random.permutation(len(X))
# X = X[idxes_rand]
# Y = Y[idxes_rand]
#
# idx_split = int(len(X) * 0.8)
# dataset_train = (X[:idx_split], Y[:idx_split])
# dataset_test = (X[idx_split:], Y[idx_split:])
#
# np.random.seed(int(time.time()))
#
# class Variable:
#     def __init__(self, value):
#         self.value: np.ndarray = value
#         self.grad: np.ndarray = np.zeros_like(value)
#
#
# class LayerLinear():
#     def __init__(self, in_features: int, out_features: int):
#         self.W = Variable(
#             value=np.random.random((out_features, in_features))
#         )
#         self.b = Variable(
#             value=np.zeros((out_features,))
#         )
#         self.x: Variable = None
#         self.output: Variable = None
#
#     def forward(self, x: Variable):
#         self.x = x
#         self.output = Variable(
#             #
#             np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
#         )
#         return self.output  # this output will be input for next function in model
#
#     def backward(self):
#         # W*x + b / d b = 0 + b^{1-1} = 1
#         # d_b = 1 * chain_rule_of_prev_d_func
#         self.b.grad = 1 * self.output.grad
#
#         # d_W = x * chain_rule_of_prev_d_func
#
#         self.W.grad = np.matmul(
#             np.expand_dims(self.output.grad, axis=2),
#             np.expand_dims(self.x.grad, axis=1),
#         )
#
#         # d_x = W * chain_rule_of_prev_d_func
#         self.x.grad = np.matmul(
#             self.W.value.transpose(),
#             self.output.grad.transpose()
#         ).transpose()
#
#
# class LeakyReLU:
#     def __init__(self):
#         self.x = None
#         self.output: Variable = None
#
#     def forward(self, x: Variable) -> Variable:
#         self.x = x
#         z = x
#         for i in range(len(x.value)):
#             for j in range(len(x.value[0])):
#                 if z.value[i, j] <= 0 :
#                     z.value[i, j] *= 0.01
#
#         self.output = z
#         return self.output
#
#
#     def backward(self):
#         self.x.grad = np.ones_like(self.x.value)
#         self.x.grad[self.x.value <= 0] = 0.01
#
# class LossMAE:
#     def __init__(self):
#         self.y: Variable = None
#         self.y_prim: Variable = None
#         self.rmse: Variable = None
#
#     def forward(self, y: Variable, y_prim: Variable) -> float:
#         self.y = y
#         self.y_prim = y_prim
#         loss = np.mean(np.abs(y.value - y_prim.value))
#         return loss
#
#     def backward(self):
#         self.y_prim.grad = (self.y_prim.value - self.y.value) / np.abs(self.y.value - self.y_prim.value)
#
#     def n_rmse(self, y: Variable, y_prim: Variable, scaler) -> float:
#         self.y = y
#         self.y_prim = y_prim
#         nrmse = scaler * np.sqrt(np.mean((y.value - y_prim.value) ** 2))
#         return nrmse
#
# class Model:
#     def __init__(self):
#         self.layers = [
#             LayerLinear(in_features=13, out_features=10),
#             LeakyReLU(),
#             LayerLinear(10, 5),
#             LeakyReLU(),
#             LayerLinear(5, 1),
#         ]
#
#     def forward(self, x: Variable):
#         out = x
#         for layer in self.layers:
#             out = layer.forward(out)
#         return out
#
#     def backward(self):
#         for layer in reversed(self.layers):
#             layer.backward()
#
#     def parameters(self):
#         variables = []
#         for layer in self.layers:
#             if isinstance(layer, LayerLinear):
#                 variables.append(layer.W)
#                 variables.append(layer.b)
#         return variables
#
#
# class OptimizerSGD:
#     def __init__(self, parameters, learning_rate):
#         self.parameters = parameters
#         self.learning_rate = learning_rate
#
#     def step(self):
#         for param in self.parameters:
#             param.value -= self.learning_rate * np.mean(param.grad, axis=0)
#
#
# model = Model()
# opt = OptimizerSGD(
#     model.parameters(),
#     LEARNING_RATE
# )
# loss_fn = LossMAE()
#
#
# losses_train = []
# losses_test = []
# nrmse_tr = []
# nrmse_test = []
#
# y_max_train = np.max(dataset_train[1])
# y_min_train = np.min(dataset_train[1])
# y_max_test = np.max(dataset_test[1])
# y_min_test = np.min(dataset_test[1])
# for epoc in range(1, 300):
#
#     for dataset in [dataset_train, dataset_test]:
#         X, Y = dataset
#         losses = []
#         nrmses = []
#
#         for idx in range(0, len(X) - BATCH_SIZE, BATCH_SIZE):
#             x = X[idx:idx + BATCH_SIZE]
#             y = Y[idx:idx + BATCH_SIZE]
#
#             y_prim = model.forward(Variable(x))
#             loss = loss_fn.forward(Variable(y), y_prim)
#
#             scaler = 1 / (y_max_test - y_min_test)
#             if dataset == dataset_train:
#                 scaler = 1 / (y_max_train - y_min_train)
#
#             nrmse = loss_fn.n_rmse(Variable(y), y_prim, scaler= scaler)
#
#
#             nrmses.append(nrmse)
#             if dataset is dataset_train:
#                 loss_fn.backward()
#                 model.backward()
#                 opt.step()
#
#             losses.append(loss)
#         if dataset is dataset_train:
#             losses_train.append(np.mean(losses))
#             nrmse_tr.append(np.mean(nrmses))
#         else:
#             nrmse_test.append(np.mean(nrmses))
#             losses_test.append(np.mean(losses))
#
#     print(f'epoch: {epoc} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]} rmse_tr: {nrmse_tr[-1]} rmse_test: {nrmse_test[-1]}')
#
# plt.subplot(2, 1, 1)
# plt.title('loss')
# plt.plot(losses_train)
# plt.plot(losses_test)
#
# plt.subplot(2, 1, 2)
# plt.title('acc')
# plt.plot(nrmse_tr)
# plt.plot(nrmse_test)
# plt.show()
#
#
#

# --------------------------------------------------------------

##### Boston Regression Pytorch
import time
import sys
import sklearn.datasets
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-6
BATCH_SIZE = 16

X, Y = sklearn.datasets.load_boston(return_X_y=True)

np_y = np.copy(Y)
Y[:] = 2 * ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])) - 0.5)
Y = np.expand_dims(Y, axis=1)

np_x = np.copy(X)
for x in range(len(X[0])):
    X[:, x] = 2 * ((np_x[:, x] - np.min(np_x[:, x])) / (np.max(np_x[:, x]) - np.min(np_x[:, x])) - 0.5)

np.random.seed(0)
idxes_rand = np.random.permutation(len(X))
X = X[idxes_rand]
Y = Y[idxes_rand]

idx_split = int(len(X) * 0.8)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])
np.random.seed(int(time.time()))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.layers1 = torch.nn.Linear(in_features=13, out_features=7)
        # self.layers2 = torch.nn.Linear(in_features=7, out_features=1)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(13, 13),
            torch.nn.Tanh(),
            torch.nn.Linear(13, 1),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim

model = Model()
opt = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)

losses_train = []
losses_test = []
nrmse_tr = []
nrmse_test = []
y_max_train = np.max(dataset_train[1])
y_min_train = np.min(dataset_train[1])
y_max_test = np.max(dataset_test[1])
y_min_test = np.min(dataset_test[1])

for epoc in range(1, 700):

    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        nrmses = []

        for idx in range(0, len(X) - BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx + BATCH_SIZE]
            y = Y[idx:idx + BATCH_SIZE]

            y_prim = model.forward(torch.FloatTensor(x))
            y = torch.FloatTensor(y)

            loss = torch.mean(torch.abs(y - y_prim))
            losses.append(loss.item())

            # y = y.detach()
            # y_prim = y_prim.detach()
            scaler = 1 / (y_max_test - y_min_test)
            if dataset == dataset_train:
                scaler = 1 / (y_max_train - y_min_train)
            nrmse = scaler * torch.sqrt(torch.mean((y.detach()-y_prim.detach())**2))

            nrmses.append(nrmse.item())

            if dataset is dataset_train:
                loss.backward()
                opt.step()

        if dataset is dataset_train:
            losses_train.append(np.mean(losses))
            nrmse_tr.append(np.mean(nrmses))
        else:
            nrmse_test.append(np.mean(nrmses))
            losses_test.append(np.mean(losses))

    print(
        f'epoch: {epoc} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]} nrmse_tr: {nrmse_tr[-1]} nrmse_test: {nrmse_test[-1]}')
plt.subplot(2, 1, 1)
plt.title('loss')
plt.plot(losses_train)
plt.plot(losses_test)

plt.subplot(2, 1, 2)
plt.title('acc')
plt.plot(nrmse_tr)
plt.plot(nrmse_test)
plt.show()
