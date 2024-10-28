import numpy as np

class Node:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return str(self.out) #Valor num del nodo
# Linear
class Linear(Node):
    def __init__(self, row, column, inicializacion=None):
        if inicializacion is not None:
            if inicializacion == "Xavier":
                self.w = np.random.randn(row, column) * np.sqrt(2.0/(row+column))
            else:
                self.w = np.random.randn(row, column) * np.sqrt(2.0/row)
        else:
            self.w = np.random.randn(row, column) * 0.01
        self.b = np.array([0.0 for _ in range(row)])

    def forward(self, x):
        self.out = np.dot(self.w, x)+self.b
        return self.out

    def backward(self):
        pass
# ReLU
class ReLU(Node):
    def forward(self, x):
        self.out = np.array([np.maximum(0, a) for a in x])
        return self.out

    def backward(self):
        pass
# Tanh
class Tanh(Node):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self):
        return 1 - self.out**2
# Softmax
class Softmax(Node):
    def forward(self, x):
        a = np.array([np.exp(i) for i in x])
        denominador = np.sum(a)
        self.out = np.array([np.exp(i)/denominador for i in x])
        return self.out

    def backward(self):
        pass
# CrossEntropy
class CrossEntropy(Node):
    def forward(self, x):
        fx, y = x
        y = int(y)
        epsilon = 1e-20
        fx = np.clip(fx, epsilon, 1-epsilon)
        self.out = -np.log(fx[y])
        return self.out

    def backward(self):
        return -1/self.out

class Sequential(Node):
    def __init__(self, *kwargs):
        self.layers = kwargs
        self.params = []
        for layer in self.layers:
            if layer.params:
                self.params.append(layer)

    def forward(self, x):
        actual_val = x
        for layer in self.layers:
            actual_val = layer(actual_val)
        return actual_val

    def __getitem__(self, i):
        return self.layers[i]
