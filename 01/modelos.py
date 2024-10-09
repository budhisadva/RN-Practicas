from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
# from matplotlib import pyplot as plt
import numpy as np

class Node:
    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return str(self.out)
#---------------------------------------
class PreActivation(Node):
    def __init__(self, weights:np.array, bias:float):
        self.w = weights
        self.b = bias
        self.out = None

    def forward(self, x:np.array):
        self.out = np.dot(x, self.w) + self.b
        return self.out

    def backward(self, grad_output:float):
        grad_input = self.w * grad_output
        return grad_input
#---------------------------
class Sigmoide(Node):
    def forward(self, z:float):
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self):
        pass

    def fit(self, X, Y, T=100, lr=0.1):
        lineal = PreActivation(weights=np.random.randn(X.shape[1]), bias=np.random.randn())
        entropia = CrossEntropy()
        for t in range(T):
            Y_predicciones = []
            for i, x in enumerate(X):
                fx = self.forward(lineal(x))
                Y_predicciones.append(fx)
                delta = lr*(fx-Y[i])
                lineal.w = lineal.w - delta*x
                lineal.b = lineal.b - delta
            if entropia( (np.array(Y_predicciones), Y) ) == 0:
                return (lineal.w, lineal.b)
        return (lineal.w, lineal.b)
# -------------------------
class CrossEntropy(Node):
    def forward(self, x:tuple):
        Y_predicciones, Y = x
        acc = 0
        for i, y in enumerate(Y):
            acc += -(y*np.log(Y_predicciones[i])+(1-y)*np.log(1-Y_predicciones[i]))
        self.out = (1/len(Y)) * acc
        return self.out

    def backward(self, x):
        y, fx = x
        if fx == 0:
            return None
        return (fx-y) / fx*(1-fx)
# ---------------------
def main():
    np.random.seed(42)
    X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=10)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    #
    logistica = Sigmoide()
    w,b = logistica.fit(x_train, y_train)
    print(w, b)


if __name__ == '__main__':
    main()
