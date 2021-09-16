import time
import sys
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt



X, Y = sklearn.datasets.fetch_california_housing(return_X_y=True)
# Y.shape (N, )
Y = np.expand_dims(Y, axis=1)


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
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output # this output will be input for next function in model

    def backward(self):
        # W*x + b / d b = 0 + b^{1-1} = 1
        # d_b = 1 * chain_rule_of_prev_d_func
        self.b.grad = 1 * self.output.grad

        # d_W = x * chain_rule_of_prev_d_func

        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )

        # d_x = W * chain_rule_of_prev_d_func
        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()

class LayerReLU:
    def __init__(self):
        self.x = None
        self.output: Variable = None

    def forward(self, x: Variable) -> Variable:
        self.x = x
        self.output = Variable(
            (x.value >= 0) * x.value
        )
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0)  * self.output.grad

class LossMAE:
    def __init__(self):
        self.y: Variable = None
        self.y_prim: Variable = None

    def forward(self, y: Variable, y_prim: Variable) -> float:
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad = (self.y_prim.value - self.y.value)/np.abs(self.y.value - self.y_prim.value)


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=8, out_features=4),
            LayerReLU(),
            LayerLinear(4,4),
            LayerReLU(),
            LayerLinear(4,1)
        ]

    def forward(self, x: Variable):
        out = x
        for layer in self.layers:
             out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
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

LEARNING_RATE = 1e-2
BATCH_SIZE = 16

model = Model()
opt = OptimizerSGD(
    model.parameters(),
    LEARNING_RATE
)
loss_fn = LossMAE()

np.random.seed(0)
idxes_rand = np.random.permutation(len(X))
X = X[idxes_rand]
Y = Y[idxes_rand]

idx_split = int(len(X) * 0.8)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))

losses_train = []
losses_test = []

for epoc in range(1,300):

        for dataset in [dataset_train, dataset_test]:
            X,Y = dataset
            losses=[]
            for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
                x = X[idx:idx + BATCH_SIZE]
                y = Y[idx:idx + BATCH_SIZE]

                y_prim = model.forward(Variable(x))
                loss = loss_fn.forward(Variable(y), y_prim)

                losses.append(loss)
                if dataset is dataset_train:
                    loss_fn.backward()
                    model.backward()
                    opt.step()
            if dataset is dataset_train:
                losses_train.append(np.mean(losses))
            else:
                losses_test.append(np.mean(losses))

        print(f'epoch: {epoc} losses_train: {losses_train[-1]} losses_test: {losses_test[-1]}')
        if epoc % 10 == 0:
            plt.plot(losses_train)
            plt.plot(losses_test)
            plt.show()