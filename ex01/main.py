import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

digits = load_digits()

print(digits.keys())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

subset = [3, 8]

X = images[np.where([t in subset for t in target])]
y = target[np.where([t in subset for t in target])]
y[y == subset[0]] = 1
y[y == subset[1]] = -1

#plt.imshow(X[0])
#plt.show()

X = X.reshape((len(X[:,0,0]), len(X[0,:,0])*len(X[0,0,:])))

print('shape X:', np.shape(X))
print('shape y:', np.shape(y))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def gradient(b, X, y):
    return np.sum((np.ones(len(y)) - sigmoid(y * np.dot(X, b)))*(-y * X.T), axis=1) / len(y)

def predict(b, X):
    ret = np.dot(X, b)
    ret[ret >= 0] = 1
    ret[ret < 0] = -1
    return ret

def zero_one_loss(y_pred, y_true):
    return np.count_nonzero( (y_pred * y_true) - 1)

def tau(tau0, gamma, t):
    print('tau', tau0 / (1 + gamma* t))
    return tau0 / (1 + gamma* t)


def gradient_descent(X, y, b0, tau, tau0, gamma, m):
    b = b0
    for t in range(m):
        b -= tau(tau0, gamma, t)*gradient(b, X, y)
        b = b.clip(-1., 1.)
    return b

def stochastic_gradient_descent(X, y, b0, tau, tau0, gamma, m):
    b = b0
    for t in range(m):
        b -= tau(tau0, gamma, t)*gradient(b, X, y)
        b = b.clip(-1., 1.)
    return b

b0 = np.random.rand(64) * 2 - 1
tau0 = 1.
gamma = .5
m = 100

print(zero_one_loss(predict(gradient_descent(X, y, b0, tau, tau0, gamma, m), X), y))

#print(gradient_descent(X, y, b0, tau, tau0, gamma, m))

