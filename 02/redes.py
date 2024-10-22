class Node():
    """Nodo super clase con funciones generales"""
    def __init__(self):
        pass
        # Agrega los parámetros necesarios

    def __call__(self, *kwargs):
        return self.forward(*kwargs)

    def __str__(self):
        return str(self.h) #Valor num del nodo

    # Agregar demas metodos

# Nodos
# Linear
# ReLU
# Tanh
# Softmax
# CrossEntropy
