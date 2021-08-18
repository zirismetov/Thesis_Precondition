import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = 0
b = 0

def linear(W, b,  x):
    return W * x + b

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def model(W, b,  x):
    return sigmoid(linear(W, b ,x)) * 20.0

Y_prim = model(W, b, X)

print(Y_prim)
