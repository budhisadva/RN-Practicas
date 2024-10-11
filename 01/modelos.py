from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
# from matplotlib import pyplot as plt
import numpy as np

class Node:
    def __init__(self):
        '''
        Constructor de nuestra clase nodo
        '''
        self.gradiente = None

    def __call__(self, *kwargs):
        return self.forward(*kwargs)

    def forward(self, *kwargs):
        raise NotImplementedError("Aquí cada sub-clase definira su propio metodo forward")

    def backward(self, *kwargs):
        raise NotImplementedError("Aquí cada subclase tendrá que implementar su metodo backward")

    def __str__(self):
        return str(self.out) # valor núm del nodo
#-----------------Aquí va nuestra función para pre-activación----------------------
class Linear(Node):
    '''
    Función que nos sirve para las pre-activaciones
    '''
    def __init__(self, input_size, output_size):
        #self.w = weights
        self.w = np.random.randn(input_size)
        self.b = np.random.randn(output_size)
        self.out = None

    def forward(self, x:np.array):
        self.out = np.dot(x, self.w) + self.b
        return self.out

    def backward(self, grad_output:float):
        grad_input = self.w * grad_output
        return grad_input
#------------Aquí van nuestras funciones de pre-activación---------------
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

class ReLU(Node):
    '''
    Función de activación RelU
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        fx = max(0,x)
        return fx

    def backward(self, grad_out):
        pass 

class Tanh(Node):
    '''
    Función de activación tangente hiperbolica
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return tanh

# ---------Aquí van las funciones para calcular el error----------------
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
