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

def main():
    x, y = make_classification()
    # print(x)
    print(y)

if __name__ == '__main__':
    main()
