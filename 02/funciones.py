from matplotlib import pyplot as plt
import numpy as np

def make_classification(r0=1, r1=3, k=1000):
    """
    Creación de los datos
    """
    X1 = [np.array([r0*np.cos(t),r0*np.sin(t)]) for t in range(0,k)]
    X2 = [np.array([r1*np.cos(t),r1*np.sin(t)]) for t in range(0,k)]
    X = np.concatenate((X1,X2))
    n,d = X.shape
    Y = np.zeros(2*k)
    Y[k:] += 1
    noise = np.array([np.random.normal(0,1,2) for i in range(n)])
    X += 0.5*noise

    return X,Y

def grafica_data(X, Y):
    plt.figure(figsize=(8,6))
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:,1], color='purple', label='Clase 0', alpha=0.5)
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:,1], color='yellow', label='Clase 1', alpha=0.5)
    plt.title('Datos generados manualmente')
    plt.legend()
    plt.show()

def crea_batch(X, size):
    n = X.shape[0]
    return [ X[i:i+size] for i in range(0, n, size)]
