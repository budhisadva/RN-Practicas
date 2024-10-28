from funciones import make_classification, grafica_data, crea_batch
from sklearn.model_selection import train_test_split
from Nodos import Linear, Tanh, ReLU, Softmax, CrossEntropy
import numpy as np

class FeedForward:
    def __init__(self, d, m1, m2, m_salida=2, capas=2):
        self.a1 = Linear(m1, d, "Xavier")
        self.h1 = Tanh()
        self.a2 = Linear(m2, m1, "He")
        self.h2 = ReLU()
        self.a3 = Linear(m_salida, m2)
        self.h3 = Softmax()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.out = self.h3(self.a3(self.h2(self.a2(self.h1(self.a1(x))))))
        return self.out

    def fit(self, X, Y, size=10, lr=0.1, T=100):
        L = CrossEntropy()
        n, d = X.shape
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        batches = crea_batch(indexes, size)
        for batch in batches:
            for i in batch:
                fx = self.forward(X[i])
                loss = L( (fx, Y[i]) )
                print(fx, int(Y[i]), loss)


def main():
    X, Y = make_classification()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    n, d = X.shape
    red = FeedForward(d, 5, 5)
    red.fit(x_train, y_train)

if __name__ == '__main__':
    main()
