import numpy as np
import tensorflow as tf

class Domain:
    def __init__(self, x1, x2, indicator, gamma, a, b):
        self.x1 = x1 # np.array
        self.x2 = x2 # np.array
        self.indicator = indicator # function defined on samples
        self.a = a
        self.b = b
        self.gamma = gamma


# def sample(x1, x2, n):
#     d = x1.shape[0]
#     x = np.zeros(shape=(n,d))
#     for i in range(n):
#         for j in range(d):
#             x[i][j] = np.random.uniform(x1[j], x2[j])
#     return x

def sample(x1, x2, n):
    d = x1.shape[0]
    U = tf.random.uniform(shape=(n,d), dtype=tf.float64)
    x = U * (x2 - x1) + x1
    return x

# def mc_int(G, f, n):
#     sides = np.abs(G.x2 - G.x1)
#     vol = np.prod(sides)
#     x = sample(G.x1, G.x2, n)
#     mask = G.indicator(x)
#     vals = f(x) * mask
#     return vol * np.mean(vals)


def mc_int(G, f, n):
    sides = tf.abs(G.x2 - G.x1)
    vol = tf.reduce_prod(sides)
    x = sample(G.x1, G.x2, n)
    mask = G.indicator(x)
    vals = f(x) * mask
    mean = tf.reduce_mean(vals)
    res = vol * mean
    return res

def check_boundary_cond(f1, f2, G, n):
    x = np.random.uniform(G.a, G.b, n)
    path = G.gamma(x)
    delta = f1(path) - f2(path)
    return (G.b - G.a) * np.mean(delta**2)
