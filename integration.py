import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Domain:
    def __init__(self, x1, x2, indicator, gamma, a, b):
        self.x1 = x1 # np.array
        self.x2 = x2 # np.array
        self.indicator = indicator # function defined on 2_dsamples
        self.a = a
        self.b = b
        self.gamma = gamma # v-function defined on 1d_samples


def sample(x1, x2, n):
    d = x1.shape[0]
    U = tf.random.uniform(shape=(n,d), dtype=tf.float32)
    x = U * (x2 - x1) + x1
    return x


def mc_int(G, f, n):
    sides = tf.cast(tf.abs(G.x2 - G.x1),dtype=tf.float32)
    vol = tf.reduce_prod(sides)
    x = sample(G.x1, G.x2, n)
    mask = G.indicator(x)
    vals = f(x) * mask
    mean = tf.reduce_mean(vals)
    res = vol * mean
    return res


def check_boundary_cond(f1, f2, G, n):
    x = tf.random.uniform(shape=(n,),
                          minval=G.a, maxval=G.b, dtype=tf.float32)
    path = G.gamma(x)
    delta = f1(path) - f2(path)
    return (G.b - G.a) * tf.reduce_mean(delta**2)