# Teaching example from page 8; fixed weights, no training.
from math import exp, tanh

def sigmoid(z):
    return 1 / (1 + exp(-z))

h, c = 0.0, 0.0
for day, x in enumerate([0.0, 1.0, 0.2], 1):
    f = sigmoid(0.3*x + 0.2*h + 1.0)
    i = sigmoid(0.8*x - 0.1*h - 0.5)
    g = tanh(0.9*x + 0.2*h)
    o = sigmoid(0.2*x + 0.1*h)
    c = f*c + i*g
    h = o*tanh(c)
    print(day, *(round(v, 4)
                 for v in [f, i, g, o, c, h]))
