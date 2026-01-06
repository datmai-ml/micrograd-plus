import random
from micrograd.engine import Value
from micrograd.activations import relu, sigmoid, tanh

class Module:

    def parameters(self):
        return[]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):

    def __init__(self, nin, nonlin=True, activation=None):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.activation = activation

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        if self.nonlin:
            if self.activation == 'ReLU':
                return relu(act)
            elif self.activation == 'sigmoid':
                return sigmoid(act)
            elif self.activation == 'tanh':
                return tanh(act)
            elif self.activation == None:
                return act
        else:
            return act
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.activation if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):

    def __init__(self, nin, nout, nonlin=True, activation=None, **kwargs):
        self.neurons = [Neuron(nin, nonlin=nonlin, activation=activation, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"
    
class MLP(Module):

    def __init__(self, nin, nouts, activation=None):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1, activation=activation if i!=len(nouts)-1 else None) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
