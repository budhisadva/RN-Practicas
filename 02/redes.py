from funciones import make_classification, grafica_data, crea_batch
from Nodos import Linear, Tanh, ReLU, Softmax, CrossEntropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class FeedForward:
    def __init__(self, d, m1, m2, m_salida=2, capas=2):
        self.a1 = Linear(m1, d, "Xavier")
        self.h1 = Tanh()
        self.a2 = Linear(m2, m1, "He")
        self.h2 = ReLU()
        self.a3 = Linear(m_salida, m2)
        self.h3 = Softmax()
        self.L = CrossEntropy()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.out = self.h3(self.a3(self.h2(self.a2(self.h1(self.a1(x))))))
        return self.out

    def backward(self, *kwargs):
        fx, y = kwargs
        self.a1.backward(self.h1.backward(self.a2.backward(self.h2.backward(self.a3.backward(self.h3.backward(fx, y))))))

    def adagrad(self, epsilon=1e-8, lr=0.1):
        self.a3.m_w = self.a3.m_w + np.pow(self.a3.grad_w,2)
        self.a3.m_b = self.a3.m_b + np.pow(self.a3.grad_b,2)
        self.a3.w -= (lr/np.sqrt(self.a3.m_w+epsilon)) * self.a3.grad_w
        self.a3.b -= (lr/np.sqrt(self.a3.m_b+epsilon)) * self.a3.grad_b
        #
        self.a2.m_w = self.a2.m_w + np.pow(self.a2.grad_w,2)
        self.a2.m_b = self.a2.m_b + np.pow(self.a2.grad_b,2)
        self.a2.w -= (lr/np.sqrt(self.a2.m_w+epsilon)) * self.a2.grad_w
        self.a2.b -= (lr/np.sqrt(self.a2.m_b+epsilon)) * self.a2.grad_b
        #
        self.a1.m_w = self.a1.m_w + np.pow(self.a1.grad_w,2)
        self.a1.m_b = self.a1.m_b + np.pow(self.a1.grad_b,2)
        self.a1.w -= (lr/np.sqrt(self.a1.m_w+epsilon)) * self.a1.grad_w
        self.a1.b -= (lr/np.sqrt(self.a1.m_b+epsilon)) * self.a1.grad_b

    def fit(self, X, Y, size=10, lr=0.1, T=100):
        n, d = X.shape
        for t in range(T):
            indexes = np.arange(n)
            np.random.shuffle(indexes)
            batches = crea_batch(indexes, size)
            for batch in batches:
                for i in batch:
                    fx = self.forward(X[i])
                    loss = self.L(fx, Y[i])
                    self.backward(fx, Y[i])
                    self.adagrad()

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.forward(x))
        return np.array(Y)

def main():
    X, Y = make_classification()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    n, d = X.shape
    red = FeedForward(d, 5, 6)
    red.fit(x_train, y_train)
    y_pred = red.predict(x_test)
    tem = []
    for i in range(len(y_test)):
        tem.append(y_pred[i][int(y_test[i])])
    lista = [1 if x > 0.5 else 0 for x in tem]
    report = classification_report(y_test, lista)
    print(report)

if __name__ == '__main__':
    main()
