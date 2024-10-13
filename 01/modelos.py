from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
# from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

class Node:
    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return str(self.out)
#---------------------------------------
class PreActivation(Node):
    def __init__(self, d:float):
        self.w = np.random.randn(d)
        self.b = np.random.randn()

    def forward(self, x:np.array):
        self.x = x
        self.out = np.dot(x, self.w) + self.b
        return self.out

    def backward(self, cadena:float):
        self.grad_w = self.x*cadena
        self.grad_b = cadena
        return self.grad_w

    def fit(self, X, Y, T=100, lr=0.1):
        L = SquaredError()
        for t in range(T):
            np.random.shuffle(X)
            grad_acc = np.zeros(len(X[0]))
            for i, x in enumerate(X):
                fx = self.forward(x)
                L((fx,Y[i]))
                grad_r = self.backward(L.backward())
                grad_acc += grad_r
            self.w -= lr*grad_acc
            self.b -= lr*self.grad_b

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.forward(x))
        return np.array(Y)
#---------------------------
class Sigmoide(Node):
    def forward(self, z:float):
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self, grad_l:float):
        self.grad_s = self.out*(1-self.out)
        return self.grad_s*grad_l

    def fit(self, X, Y, T=100, lr=0.1):
        pre = PreActivation(X.shape[1])
        L = CrossEntropy()
        for t in range(T):
            np.random.shuffle(X)
            grad_acc = np.zeros(len(X[0]))
            for i, x in enumerate(X):
                fx = self.forward(pre(x))
                L((fx,Y[i]))
                grad_r = pre.backward(self.backward(L.backward()))
                grad_acc += grad_r
            pre.w -= lr*grad_acc
            pre.b -= lr*pre.grad_b
        self.z = pre

    def predict(self, X):
        Y = []
        perceptron = Perceptron()
        for x in X:
            Y.append(perceptron(self.forward(self.z(x))))
        return np.array(Y)
# -------------------------
class Perceptron(Node):
    def forward(self, x:float):
        if x > 0.5:
            self.out = 1
        else:
            self.out = 0
        return self.out
# -------------------------
class CrossEntropy(Node):
    def forward(self, x:tuple):
        fx, y = x
        epsilon = 1e-15
        fx = np.clip(fx, epsilon, 1-epsilon)
        self.out = -(y*np.log(fx)+(1-y)*np.log(1-fx))
        if y == 0:
            self.grad_l = 1/(1-fx)
        else:
            self.grad_l = -(1/fx)

    def backward(self):
        return self.grad_l
# ---------------------
class SquaredError(Node):
    def forward(self, x:tuple):
        fx, y = x
        epsilon = 1e-18
        delta = y-fx
        delta = np.clip(delta, epsilon, 1-epsilon)
        self.out = pow(delta,2)
        self.grad_l = (delta)*-2

    def backward(self):
        return self.grad_l
# ---------------------
def entrena_modelo_logistico():
    X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=10)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    modelo = Sigmoide()
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(report)

def entrena_modelo_lineal():
    X, Y = make_regression(n_samples=1000, n_features=2, n_informative=2)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    modelo = PreActivation(X.shape[1])
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse}")
    print(f"R2: {r2}")

def main():
    np.random.seed()
    # entrena_modelo_logistico()
    entrena_modelo_lineal()

if __name__ == '__main__':
    main()
