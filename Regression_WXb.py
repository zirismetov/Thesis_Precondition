import numpy as np

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = 0
b = 0

def linear(W, b, x):
    return W * x + b

def dW_linear(W, b, x):
    return 0 #TODO

def db_linear(W, b, x):
    return 0 #TODO

def dx_linear(W, b, x):
    return 0 #TODO

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def da_sigmoid(a):
    return np.exp(-a)/(1 + np.exp(-a))**2

def model(W, b, x):
    return sigmoid(linear(W, b, x)) * 20.0

def dW_model(W, b, x):
    return da_sigmoid(linear(W, b, x)) * x * 20.0

def db_model(W, b, x):
    return da_sigmoid(linear(W, b, x)) * 20.0

def loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)

def dW_loss(y, y_prim, W, b, x): # derivative WRT Loss function
    return 2*(y-y_prim) * -(dW_model(W,b,x))

def db_loss(y, y_prim, W, b, x):
    return 2*(y-y_prim) * -(db_model(W,b,x))


learning_rate = 1e-2
count = 0
epoc = True
while epoc:
    Y_prim = model(W, b, X)
    loss1 = loss(Y, Y_prim)

    dW_loss1 = dW_loss(Y, Y_prim, W, b, X)
    db_loss1 = db_loss(Y, Y_prim, W, b, X)

    W -= dW_loss1 * learning_rate
    b -= db_loss1 * learning_rate

    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss1}')
    print(f'rounds: {count}')
    count +=1

    if loss1 <= 0.4:
        epoc = False
