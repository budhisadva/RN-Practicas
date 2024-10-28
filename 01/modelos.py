import numpy as np

class Node:
    def __init__(self, parametros = False):
        '''
        Constructor de nuestra clase nodo
        '''
        self.parametros = parametros
        self.parent = None
        self.gradiente = None
        self.out = None

    def __call__(self, *kwargs):
        return self.forward(*kwargs)


    def __str__(self):
        return str(self.out) # valor núm del nodo

    def zero_grad(self):
        self.gradiente = 0
        #self.parent = None

class Variable(Node):
    def __init__(self, out, parent=None):
        super().__init__(parametros=True)
        self.out = out
        self.parent = parent

    def backward(self, grad=1):
        self.gradiente = grad
        if self.parent is not None:
            self.parent.backward(self.gradiente)
#-----------------Aquí va nuestra función para pre-activación----------------------
class Linear(Node):
    '''
    Función que nos sirve para las pre-activaciones
    '''
    def __init__(self, input_size, output_size):
        super().__init__(parametros=True)
        # el input_size deben ser la cantidad de neuronas de la capa actual
        # mientras que el output_size son la cantidad de neuronas en la siguiente capa
        #np.random.seed(42)
        self.w = np.random.randn(input_size, output_size) #al hacer (input_size x output_size) obtenemos una matriz de dimensiones (input_size, output_size)
        self.b = np.random.randn(output_size)
        #self.out = None

    def forward(self, x):
        self.parent = x
        self.out = np.dot(x.out, self.w) + self.b
        return Variable(self.out, parent=self)

    def backward(self, grad_output:float):
        # gradientes para los pesos y el bias
        self.grad_w = np.dot(self.parent.out.T, grad_output) 
        self.grad_b = np.mean(grad_output, axis=0)

        self.gradiente = np.dot(grad_output, self.w.T)

        # llama el backward del nodo padre para la propagacion
        if self.parent is not None:
            self.parent.backward(self.gradiente)
        

#------------Aquí van nuestras funciones de pre-activación---------------
class Sigmoide(Node):
    def forward(self, z:float):
        self.parent = z
        self.out = 1 / (1 + np.exp(-z.out))
        return Variable(self.out, parent=self)

    def backward(self, grad=1):
        grad_in = self.out * (1 - self.out) # derivada de la sigmoide
        self.gradiente = grad_in * grad #g gradiente local
        if self.parent is not None:
            self.parent.backward(grad=self.gradiente) # propagacion hacia atras si existe el padre

class ReLU(Node):
    '''
    Función de activación RelU
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.parent = x
        self.out = np.maximum(0,x.out) 
        return Variable(self.out, parent=self)

    def backward(self, grad=1):
        relu_grad = (self.parent.out > 0).astype(float)
        self.gradiente = grad * relu_grad
        self.parent.backward(grad=self.gradiente)

class Tanh(Node):
    '''
    Función de activación tangente hiperbolica
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.parent = x
        self.out = (np.exp(x.out) - np.exp(-x.out)) / (np.exp(x.out) + np.exp(-x.out))
        return Variable(self.out, parent=self)

    def backward(self, grad=1):
        tanh_grad = 1 - self.out**2
        self.gradiente = tanh_grad*grad
        self.parent.backward(grad=self.gradiente)

class Softmax(Node):
    '''
    activación softmax
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.parent = x
        exps = np.exp(x.out - np.max(x.out, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return Variable(self.out, parent=self)

    def backward(self, grad=1):
        #al parecer todo se simplifica con
        #cross entropy y softmax
        self.gradiente = grad
        if self.parent is not None:
            self.parent.backward(grad=self.gradiente)


# ---------Aquí van las funciones para calcular el error----------------
class CrossEntropy(Node):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        self.parent = y_pred
        self.y_true = y_true.reshape(-1,1) #parece que no es necesario un reshape para multiclase

        epsilon = 1e-15 # para evitar caer en un log(0)
        self.y_pred = np.clip(y_pred.out, epsilon, 1-epsilon)

        self.out = -np.sum(self.y_true*np.log(self.y_pred)) / self.y_true.shape[0]
        return Variable(self.out, parent=self)

    def backward(self, grad=1):
        #epsilon = 1e-15
        #y_pred = self.parent.out
        self.res = ((self.y_pred - self.y_true) / self.y_true.shape[0]) * grad
        if self.parent is not None:
            self.parent.backward(grad=self.res)
# ---------------------


if __name__ == '__main__':
    main()
