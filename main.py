import tensorflow as tf
import numpy as np
# from diff import *
from integration import *

def diff_test1():
    x=[1.2, -2.3]

    def f(x):
        return x[0]**3 * x[1]**4

    def df(x):
        return [3 * x[0]**2 * x[1]**4, 4 *x[0]**3 * x[1]**3]

    def ddf(x):
        return [6 * x[0] * x[1]**4, 12 * x[0]**3 * x[1]**2]

    print(diff(f,x))
    print(df(x))
    print(diff2(f,x))
    print(ddf(x))


def sample_test():
    x1 = np.array([10.0, -31.0, -2])
    x2 = np.array([15.0, -19,  -4])
    print(x2-x1)
    s = sample(x1, x2, 3)
    print(s)


sample_test()

tf.a