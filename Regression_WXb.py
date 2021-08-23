import matplotlib.pyplot as plt
# # ------------------------------Task from workshop #4
# import numpy as np
#
# X = np.array([1, 2, 3, 5])
# Y = np.array([0.7, 1.5, 4.5, 9.5])
#
# W = 0
# b = 0
#
# def linear(W, b, x):
#     return W * x + b
#
# def dW_linear(W, b, x):
#     return x
#
# def db_linear(W, b, x):
#     return 1
#
# def dx_linear(W, b, x):
#     return W
#
# def sigmoid(a):
#     return 1 / (1 + np.exp(-a))
#
# def da_sigmoid(a):
#     return 1/(1+np.exp(-a)) * (1 - 1/(1+np.exp(-a)))
#
# def model(W, b, x):
#     return sigmoid(linear(W, b, x)) * 20.0
#
# def dW_model(W, b, x):
#     return da_sigmoid(linear(W, b, x)) * x * 20.0
#
# def db_model(W, b, x):
#     return da_sigmoid(linear(W, b, x)) * 20.0 * 1
#
# def loss(y, y_prim):
#     return np.mean((y - y_prim) ** 2)
#
# def dW_loss(y, y_prim, W, b, x): # derivative WRT Loss function
#     return 2*(y-y_prim) * -(dW_model(W,b,x))
#
# def db_loss(y, y_prim, W, b, x):
#     return 2*(y-y_prim) * -(db_model(W,b,x))
#
#
# learning_rate = 1e-2
# count = 0
# epoc = True
# while epoc:
#     Y_prim = model(W, b, X)
#     loss1 = loss(Y, Y_prim)
#
#     dW_loss1 = dW_loss(Y, Y_prim, W, b, X)
#     db_loss1 = db_loss(Y, Y_prim, W, b, X)
#
#     W -= dW_loss1 * learning_rate
#     b -= db_loss1 * learning_rate
#
#     print(f'Y_prim: {Y_prim}')
#     print(f'loss: {loss1}')
#     print(f'rounds: {count}')
#     count +=1
#
#     if loss1 <= 0.4:
#         epoc = False


# # ---------------------------------------Task 4 from HW
#
# X = np.array([1, 2, 3, 5])
# Y = np.array([0.7, 1.5, 4.5, 9.5])
#
# W1 = np.random.random()
# W2 = np.random.random()
# b1 = 0
# b2 = 0
#
#
# def linear(W, b, x):
#     a = W * x + b
#     return a
#
#
# def dW_linear(W, b, x):
#     return W * x + b
#
#
# def db_linear(W, b, x):
#     return W * x + b
#
#
# def relu(a):
#     z = np.array(a)
#     z[z < 0] = 0
#     return z
#
#
# def drelu(a):
#     z = np.array(a)
#     z[z >= 0] = 1
#     z[z < 0] = 0
#     return z
#
#
# def model(w1, w2, b1, b2, x):
#     return linear(w2, b2, relu(linear(w1, b1, x)))
#
#
# def loss(y, y_prim):
#     return np.mean(np.abs(y - y_prim))
#
#
# def dW1_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(w2 * drelu(linear(w1, b1, x)) * x)
#
#
# def dW2_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(drelu(linear(w1, b1, x)))
#
#
# def db1_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(w2 * drelu(linear(w1, b1, x)))
#
#
# def db2_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(1)
#
#
# learning_rate = 1e-2
#
# losses = []
# for epoc in range(200):
#     Y_prim = model(W1, W2, b1, b2, X)
#     loss1 = loss(Y, Y_prim)
#     losses.append(loss1)
#
#     dw1loss = np.mean(dW1_loss(Y, Y_prim, W1, b1, W2, b2, X))
#     dw2loss = np.mean(dW2_loss(Y, Y_prim, W1, b1, W2, b2, X))
#
#     db1loss = np.mean(db1_loss(Y, Y_prim, W1, b1, W2, b2, X))
#     db2loss = np.mean(db2_loss(Y, Y_prim, W1, b1, W2, b2, X))
#
#     W1 -= dw1loss * learning_rate
#     W2 -= dw2loss * learning_rate
#
#     b1 -= db1loss * learning_rate
#     b2 -= db2loss * learning_rate
#
#     print(f'loss: {loss1}')
#     print(f'Y_prim: {Y_prim}')
# plt.plot(losses)
# plt.title("loss")
# plt.show()

#  -----------------------------------Task 5 from HW

import numpy as np

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = np.random.random()
b = 0


def linear(W, b, x):
    a = W * x + b
    return a


def dW_linear(W, b, x):
    return x


def db_linear(W, b, x):
    return 1


def dx_linear(W, b, x):
    return tanh(linear(W, x, b))


def tanh(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))


def da_tanh(a):
    # return
    return 1 - tanh(a)**2


def LeakyReLU(z):
    z = np.array(z)
    for i in range(len(z)):
        if z[i] <= 0:
            z[i] *= learning_rate
    return z

def der_LeakyReLU(z):
    dz = np.ones_like(z)
    dz[z<0] = learning_rate
    return dz

def model(w, b, x):
    return LeakyReLU(linear(w, b, tanh(linear(w, b, x))))


def dW_model(W, b, x):
    return der_LeakyReLU(linear(W, b, tanh(linear(W, b, x)))) * W * da_tanh(linear(W,b,x)) * dW_linear(W, b, x)

def db_model(W, b, x):
    return der_LeakyReLU(linear(W, b, tanh(linear(W, b, x)))) * W * da_tanh(linear(W,b,x)) * db_linear(W, b, x)

def loss(y, y_prim):
    return np.mean(np.abs(y - y_prim))


def dW_loss(y, y_prim, W, b, x):  # derivative WRT Loss function
    return (y - y_prim) / np.abs(y - y_prim) * -(dW_model(W, b, x))


def db_loss(y, y_prim, W, b, x):
    return (y - y_prim) / np.abs(y - y_prim) * -(db_model(W, b, x))


learning_rate = 0.01
losses = []
epoc = True
for i in range(30000):
    Y_prim = model(W, b, X)
    loss1 = loss(Y, Y_prim)
    losses.append(loss1)

    dW_loss1 = np.mean(dW_loss(Y, Y_prim, W, b, X))
    db_loss1 = np.mean(db_loss(Y, Y_prim, W, b, X))

    W -= dW_loss1 * learning_rate
    b -= db_loss1 * learning_rate

    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss1}')
    if loss1 <= 1:
        epoc = False
plt.plot(losses)
plt.title("losses")
plt.show()