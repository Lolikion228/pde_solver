import numpy as np

class Domain:
    def __init__(self, x1, x2, indicator, gamma):
        # what if x1[i] > x2[i] for some i?
        self.x1 = x1
        self.x2 = x2
        self.indicator = indicator
        self.gamma = gamma


def sample(x1, x2, n):
    d = x1.shape[0]
    x = np.zeros(shape=(n,d))
    for i in range(n):
        for j in range(d):
            x[i][j] = np.random.uniform(x1[j], x2[j])
    return x


def mc_int(G, f, n):
    sides = np.abs(G.x2 - G.x1)
    vol = np.prod(sides)
    x = sample(G.x1, G.x2, n)
    mask = G.indicator(x)
    vals = f(x) * mask
    return vol * np.mean(vals)
