import numpy as np
from modelos import Linear

class AdagradOptimizer():
    def __init__(self, learning_rate=0.1, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_squared_accum = {}

    def get_params_and_grads(self, model):
        params, grads = {}, {}
        linear_index = 1

        for layer in model.params:
            if isinstance(layer, Linear):
                params[f"w{linear_index}"] = layer.w
                params[f"b{linear_index}"] = layer.b
                grads[f"w{linear_index}"] = layer.grad_w
                grads[f"b{linear_index}"] = layer.grad_b
                linear_index += 1

        return params, grads

    def apply_updates(self, model, params):
        linear_index = 1
        for i, layer in enumerate(model.params):
            if isinstance(layer, Linear):
                layer.w = params[f"w{linear_index}"]
                layer.b = params[f"b{linear_index}"]
                linear_index += 1

    def update(self, model):
        #obtenemos los parametros y gradientes del modelo
        params, grads = self.get_params_and_grads(model)
        # si es la primeras vez, inicializamos el acumulador de gradientes
        for key in params:
            if key not in self.grad_squared_accum:
                self.grad_squared_accum[key] = np.zeros_like(grads[key])
            # acumulamos el cuadrado de los gradientes
            self.grad_squared_accum[key] = self.grad_squared_accum[key] + (grads[key])**2
            # actualizamos los parametros usando Adagrad
            params[key] = params[key] - ((self.learning_rate) / (np.sqrt(self.grad_squared_accum[key]) + self.epsilon)) * grads[key]

        # aplicamos los parametros actualizados al modelo
        self.apply_updates(model, params)