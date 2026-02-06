from integration import *
from diff import *
import tensorflow as tf


"""
TODO
1) make vectorized ver of compute_L
2) check types in compute_loss
"""

eps = 1e-4
N = 10000

class Problem:

    def __init__(self, G, g, coefs):
        self.G = G
        self.g = g #func for bound_cond
        self.coefs = coefs
        
    
    def compute_L(self, f, x):
        L = tf.Variable(0.0, dtype=tf.float64)

        df = diff(f, x)
        ddf = diff2(f, x)

        if "xx" in self.coefs.keys():
            L = L + self.coefs["xx"](x) * ddf[0]
        if "yy" in self.coefs.keys():
            L += self.coefs["yy"](x) * ddf[1]
        if "x" in self.coefs.keys():
            L += self.coefs["x"](x) * df[0]
        if "y" in self.coefs.keys():
            L += self.coefs["y"](x) * df[1]
        if "_" in self.coefs.keys():
            L += self.coefs["_"](x) * f(x)

        return L

    
    def compute_loss(self, h):

        def integrand(x):
            vals = []
            for i in range(x.shape[0]):
                vals.append(self.compute_L(h,x[i,:])**2)
            res = tf.stack(vals)
            return res
        
        main_loss = mc_int(self.G, integrand, 100)
        # boundary_loss = check_boundary_cond(h, self.g, self.G, N)
        return 0.5 * ( main_loss + 0.0 )
        
