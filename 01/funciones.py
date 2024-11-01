import numpy as np
import matplotlib.pyplot as plt
from modelos import Variable

def draw_regions(f, x, y='black'):
    figure = plt.figure()
    min1, max1 = x[:, 0].min()-1, x[:, 0].max()+1
    min2, max2 = x[:, 1].min()-1, x[:, 1].max()+1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = Variable(np.hstack((r1,r2)))
    yhat = f(grid).argmax()
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.6)

    plt.scatter(x[:,0], x[:,1],c=y,s=2)
    return figure

def make_classification(r0=1, r1=3, k=1000):
    """
    funcion para creacion de nuestros datos para problemas de clasificacion
    """
    X1 = [np.array([r0*np.cos(t), r0*np.sin(t)]) for t in range(0,k)]
    X2 = [np.array([r1*np.cos(t), r1*np.sin(t)]) for t in range(0,k)]

    X = np.concatenate((X1,X2))
    n, d = X.shape
    Y = np.zeros(2*k)
    Y[k:] += 1
    noise = np.array([np.random.normal(0,1,2) for i in range(n)])
    X += 0.5*noise

    return X, Y
