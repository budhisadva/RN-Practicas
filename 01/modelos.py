from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import numpy as np

class Node:
    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return str(self.out)

class PreActivation(Node):
    def __init__(self, d:float):
        self.w = np.random.randn(d)
        self.b = np.random.randn()

    def forward(self, x:np.array):
        self.out = np.dot(x, self.w) + self.b
        self.x = x
        return self.out

    def backward(self, cadena:float):
        return self.x*cadena

    def fit(self, X, Y, T=100, lr=0.1):
        n, d = X.shape
        L = SquaredError()
        for t in range(T):
            grad_w = np.zeros(d)
            grad_b = []
            loss = []
            indexes = np.arange(n)
            np.random.shuffle(indexes)
            for i in indexes:
                z = self.forward(X[i])
                loss.append(L((z,Y[i])))
                parcial = L.backward((z,Y[i]))
                grad_w += self.backward(parcial)
                grad_b.append(parcial)
            R = np.mean(loss)
            grad_b_R = np.mean(grad_b)
            grad_w_R = (1/len(X))*grad_w
            if R == 0:
                return
            self.w -= lr*grad_w_R
            self.b -= lr*grad_b_R

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.forward(x))
        return np.array(Y)

class Sigmoide(Node):
    def forward(self, z:float):
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def fit(self, X, Y, T=100, lr=0.1):
        n, d = X.shape
        linear = PreActivation(d)
        L = CrossEntropy()
        for t in range(T):
            grad_w = np.zeros(d)
            grad_b = []
            loss = []
            indexes = np.arange(n)
            np.random.shuffle(indexes)
            for i in indexes:
                fx = self.forward(linear(X[i]))
                loss.append(L((fx, Y[i])))
                delta = L.backward((fx,Y[i]))
                grad_w += linear.backward(delta)
                grad_b.append(delta)
            R = np.mean(loss)
            grad_b_R = np.mean(grad_b)
            grad_w_R = (1/len(X))*grad_w
            if R == 0:
                return
            linear.w -= lr*grad_w_R
            linear.b -= lr*grad_b_R
        self.z = linear

    def predict(self, X):
        Y = []
        perceptron = Perceptron()
        for x in X:
            Y.append(perceptron(self.forward(self.z(x))))
        return np.array(Y)

class Perceptron(Node):
    def forward(self, x:float):
        if x > 0.5:
            self.out = 1
        else:
            self.out = 0
        return self.out

    def fit(self, X, Y, T=100, lr=0.1):
        n, d = X.shape
        linear = PreActivation(d)
        linear.w = np.array([ 1.0 for x in range(d)])
        linear.b = 1
        logistica = Sigmoide()
        L = SquaredError()
        for t in range(T):
            grad_w = np.zeros(d)
            grad_b = []
            loss = []
            indexes = np.arange(n)
            np.random.shuffle(indexes)
            for i in indexes:
                fx = self.forward(logistica(linear(X[i])))
                loss.append(L((fx,Y[i])))
                delta = fx-Y[i]
                grad_w += linear.backward(delta)
                grad_b.append(delta)
            grad_b_R = np.mean(grad_b)
            grad_w_R = (1/len(X))*grad_w
            R = np.mean(loss)
            if R == 0:
                self.z = linear
                return
            linear.w -= lr*grad_w_R
            linear.b -= lr*grad_b_R
        self.z = linear

    def predict(self, X):
        Y = []
        log = Sigmoide()
        for x in X:
            Y.append(self.forward(log(self.z(x))))
        return np.array(Y)


class CrossEntropy(Node):
    def forward(self, x:tuple):
        fx, y = x
        epsilon = 1e-20
        fx = np.clip(fx, epsilon, 1-epsilon)
        self.out = -(y*np.log(fx)+(1-y)*np.log(1-fx))
        return self.out

    def backward(self, x:tuple):
        fx, y = x
        return fx-y

class SquaredError(Node):
    def forward(self, x:tuple):
        fx, y = x
        self.out = np.square(y-fx)
        return self.out

    def backward(self, x:tuple):
        fx, y = x
        return -2*(y-fx)

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

def entrena_perceptron():
    #
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    print("->")
    Y1 = np.array([1,1,0,1])
    implicacion = Perceptron()
    implicacion.fit(X, Y1, 100, 1)
    y_pred = implicacion.predict(X)
    print(implicacion.z.w, implicacion.z.b)
    #
    print("NAND")
    Y2 = np.array([1,1,1,0])
    NAND = Perceptron()
    NAND.fit(X, Y2, 100, 1)
    y_pred = NAND.predict(X)
    print(NAND.z.w, NAND.z.b)
    #
    print("NOR")
    Y3 = np.array([1,0,0,0])
    NOR = Perceptron()
    NOR.fit(X, Y3, 100, 1)
    y_pred = NOR.predict(X)
    print(NOR.z.w, NOR.z.b)
    #
    print("Pregunta 7")
    X4 = np.array([[1,1,1], [1,0,1], [0,1,1], [1,0,0], [0,1,0], [0,0,0]])
    Y4 = np.array([1,1,1,0,0,0])
    modelo = Perceptron()
    modelo.fit(X4, Y4, 100, 0.5)
    print(modelo.z.w, modelo.z.b)

def main():
    # np.random.seed(42)
    # entrena_modelo_logistico()
    # entrena_modelo_lineal()
    entrena_perceptron()

if __name__ == '__main__':
    main()
