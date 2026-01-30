import numpy as np 

def df(f, x, h):
    xr = x+h
    xl = x-h
    r = f(xr)
    l = f(xl)
    return (r - l) / (2 * h)


def pdf(f, x, i, h):

    def g(t):
        x2 = np.copy(x)
        x2[i] = t
        return f(x2)
    
    return df(g, x[i], h)


def ddf(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def pddf(f, x, i, h):

    def g(t):
        x2 = x
        x2[i] = t
        return f(x2)
    
    return ddf(g, x[i], h)