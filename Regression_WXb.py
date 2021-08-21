# # Task 1
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
#     return 0 #TODO
#
# def db_linear(W, b, x):
#     return 0 #TODO
#
# def dx_linear(W, b, x):
#     return 0 #TODO
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


# ---------------------------------------Task 2
# import numpy as np
#
# X = np.array([1, 2, 3, 5])
# Y = np.array([0.7, 1.5, 4.5, 9.5])
#
# W1 = 0
# W2 = 0
# b1 = 0
# b2 = 0
#
#
# def linear(W, b, x):
#      a = W * x + b
#      return a
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
#     z[z >= 0] = 1
#     z[z < 0] = 0
#     return z
#
# def model(w1, w2, b1, b2, x):
#
#     return w2 * relu(linear(w1,b1,x)) + b2
#
#
# def loss(y, y_prim):
#     return np.mean(np.abs(y - y_prim))
#
#
# def dW1_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(w2 * relu(linear(w1, b1, x)) * x)
#
#
# def dW2_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(relu(linear(w1, b1, x)))
#
#
# def db1_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(w2 * relu(linear(w1, b1, x)))
#
#
# def db2_loss(y, y_prim, w1, b1, w2, b2, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(1)
#
#
# learning_rate = 1e-2
#
# round = 0
# epoc = True
# while epoc:
#     Y_prim = model(W1, W2, b1, b2, X)
#     loss1 = loss(Y, Y_prim)
#
#     db1loss = np.mean(db1_loss(Y, Y_prim, W1, b1, W2, b2, X))
#     db2loss = np.mean(db2_loss(Y, Y_prim, W1, b1, W2, b2, X))
#
#     dw1loss = np.mean(dW1_loss(Y, Y_prim, W1, b1, W2, b2, X))
#     dw2loss = np.mean(dW2_loss(Y, Y_prim, W1, b1, W2, b2, X))
#
#     W1 -= dw1loss * learning_rate
#     W2 -= dw2loss * learning_rate
#
#     b1 -= db1loss * learning_rate
#     b2 -= db2loss * learning_rate
#
#     round += 1
#     print(f'loss: {loss1}')
#     print(f'Y_prim: {Y_prim}')
#     print(f'rounds: {round}')
#     if loss1 <= 2.95:
#         print(W1, W2, b1, b2)
#         epoc = False


# #  -----------------------------------Task 3
#
# import numpy as np
#
# X = np.array([1, 2, 3, 5])
# Y = np.array([0.7, 1.5, 4.5, 9.5])
#
# W = 0
# b = 0
#
#
# def linear(W, b, x):
#     a = W * x + b
#     return a
#
#
# def dW_linear(W, b, x):
#     return x
#
#
# def db_linear(W, b, x):
#     return 1
#
#
# def dx_linear(W, b, x):
#     return tanh(linear(W, x, b))
#
#
# def tanh(a):
#     return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
#
#
# def da_tanh(a):
#     return (4 * np.exp(2*a)) / (np.exp(2*a) + 1) ** 2
#
#
# def LeakyReLU(z):
#     z = np.array(z)
#     for i in range(len(z)):
#         if z[i] <= 0:
#             z[i] *= learning_rate
#     return z
#
# def model(w, b, x):
#     return LeakyReLU(linear(w, b, tanh(linear(w, b, x))))
#
#
# def dW_model(W, b, x):
#     return LeakyReLU(linear(W, b, tanh(linear(W, b, x)))) * W * da_tanh(linear(W,b,x)) * dW_linear(W, b, x)
#
# def db_model(W, b, x):
#     return LeakyReLU(linear(W, b, tanh(linear(W, b, x)))) * W * da_tanh(linear(W,b,x)) * db_linear(W, b, x)
#
# def loss(y, y_prim):
#     return np.mean(np.abs(y - y_prim))
#
#
# def dW_loss(y, y_prim, W, b, x):  # derivative WRT Loss function
#     return (y - y_prim) / np.abs(y - y_prim) * -(dW_model(W, b, x))
#
#
# def db_loss(y, y_prim, W, b, x):
#     return (y - y_prim) / np.abs(y - y_prim) * -(db_model(W, b, x))
#
#
# learning_rate = 1e-2
# for epoc in range(50):
#     Y_prim = model(W, b, X)
#     loss1 = loss(Y, Y_prim)
#
#     dW_loss1 = np.mean(dW_loss(Y, Y_prim, W, b, X))
#     db_loss1 = np.mean(db_loss(Y, Y_prim, W, b, X))
#
#     W -= dW_loss1 * learning_rate
#     b -= db_loss1 * learning_rate
#
#     print(f'Y_prim: {Y_prim}')
#     print(f'loss: {loss1}')
