from micrograd.engine import Value
import math

def relu(x: Value) -> Value:
    out = Value(x.data if x.data > 0 else 0, (x,), 'ReLU')
    
    def _backward():
        x.grad += (out.data > 0) * out.grad # gradient is either 0 or 1 depending on the input
    out._backward = _backward

    return out

def sigmoid(x: Value) -> Value:
    s = 1 / (1 + math.exp(-x.data))
    out = Value(s, (x,), 'sigmoid')

    def _backward():
        x.grad += s * (1 - s) * out.grad
    out._backward = _backward
    
    return out

def tanh(x: Value) -> Value:
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (x,), 'tanh')

    def _backward():
        x.grad = (1 - t**2) * out.grad
    out._backward = _backward

    return out