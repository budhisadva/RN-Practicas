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

    def argmax(self):
        if self.out is not None:
            return np.argmax(self.out, axis=1)
        else:
            raise ValueError("No se ha ejecutado forward, no hay salida para calcular argmax.")

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
    def __init__(self, input_size, output_size, init_type=None):
        super().__init__(parametros=True)
        
        if init_type == 'Xavier':
            self.w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        elif init_type == 'He':
            self.w = np.random.rand(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.w = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros(output_size)

    def forward(self, x):
        self.parent = x
        self.out = np.dot(x.out, self.w) + self.b
        return Variable(self.out, parent=self)

    def backward(self, grad_output:float):
        # gradientes para los pesos y el bias
        self.grad_w = np.dot(self.parent.out.T, grad_output) / self.parent.out.shape[0]
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
        super().__init__(parametros=True)

    def forward(self, x):
        self.parent = x
        self.out = np.maximum(0,x.out) 
        return Variable(self.out, parent=self)

    def backward(self, grad=1):
        #relu_grad = (self.parent.out > 0).astype(float)
        relu_grad = (1 / np.abs(self.parent.out))*np.maximum(0,self.parent.out)
        self.gradiente = grad * relu_grad
        self.parent.backward(grad=self.gradiente)

class Tanh(Node):
    '''
    Función de activación tangente hiperbolica
    '''
    def __init__(self):
        super().__init__(parametros=True)

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
        super().__init__(parametros=True)

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
        self.res = ((self.y_pred - self.y_true) / self.y_true.shape[0])
        if self.parent is not None:
            self.parent.backward(grad=self.res)
# --------------------- Aquí van los optimizadores --------------------------------------


#----------------------- class sequential ---------------------------------------------------
class Sequential(Node):
    """
    clase para secuencializar capas
    """
    def __init__(self, *kwargs):
        self.layers = kwargs
        self.params = []
        for layer in self.layers:
            if layer.parametros:
                self.params.append(layer)

    def forward(self, x):
        actual_val = x
        for layer in self.layers:
            actual_val = layer(actual_val)
        return actual_val

    def __getitem__(self, i):
        return self.layers[i]


if __name__ == '__main__':
    main()
