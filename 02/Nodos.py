import numpy as np

class Node:
    def __init__(self):
        pass

    def __call__(self, *kwargs):
        return self.forward(*kwargs)

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

    def forward(self, *kwargs):
        self.x = kwargs[0]
        self.out = np.dot(self.w, self.x)+self.b
        return self.out

    def backward(self, *kwargs):
        d = kwargs[0]
        self.grad_b = d
        self.m_w = 0
        self.m_b = 0
        self.grad_w = np.outer(d, self.x)
        return np.dot(d, self.w)
# ReLU
class ReLU(Node):
    def forward(self, *kwargs):
        x = kwargs[0]
        l = []
        for a in x:
            if a == 0:
                a = 1e-300
            l.append((1/np.abs(a))*np.maximum(0, a))
        self.derivada = l
        self.out = np.array([np.maximum(0, a) for a in x])
        return self.out

    def backward(self, *kwargs):
        cadena = kwargs[0]
        self.d = self.derivada*cadena
        return self.d
# Tanh
class Tanh(Node):
    def forward(self, *kwargs):
        x = kwargs[0]
        self.out = np.tanh(x)
        return self.out

    def backward(self, *kwargs):
        cadena = kwargs[0]
        derivada = 1 - self.out**2
        self.d = derivada*cadena
        return self.d
# Softmax
class Softmax(Node):
    def forward(self, *kwargs):
        x = kwargs[0]
        max_x = np.max(x)
        denominador = np.sum([np.exp(i-max_x) for i in x])
        l = []
        for i in x:
            l.append(np.exp(i-max_x)/denominador)
        self.out = np.array(l)
        return self.out

    def backward(self, *kwargs):
        fx, y = kwargs
        self.d = fx-y
        return self.d
# CrossEntropy
class CrossEntropy(Node):
    def forward(self, *kwargs):
        fx, y = kwargs
        y = int(y)
        epsilon = 1e-20
        fx = np.clip(fx, epsilon, 1-epsilon)
        self.out = -np.log(fx[y])
        return self.out
