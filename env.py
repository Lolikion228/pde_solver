from integration import *
from diff import *
import tensorflow as tf



N = 10_000

class Problem:

    def __init__(self, G, g, coefs):
        self.G = G
        self.g = g #func for bound_cond
        self.coefs = coefs
        
    
    def compute_L(self, f, x):
        L = tf.zeros(shape=x.shape[0], dtype=tf.float32)

        df = diff(f, x)
        ddf = diff2(f, x)

        if "xx" in self.coefs.keys():
            L = L + self.coefs["xx"](x) * ddf[:,0]
        if "yy" in self.coefs.keys():
            L = L + self.coefs["yy"](x) * ddf[:,1]
        if "x" in self.coefs.keys():
            L = L + self.coefs["x"](x) * df[:,0]
        if "y" in self.coefs.keys():
            L = L + self.coefs["y"](x) * df[:,1]
        if "_" in self.coefs.keys():
            L = L + self.coefs["_"](x) * f(x)

        return L

    
    def compute_loss(self, h):

        def integrand(x):
            # x shape: (N, 2)
            return self.compute_L(h, x)**2  # shape: (N,)
        
        main_loss = integrate_over_domain(self.G, integrand, 100)
        boundary_loss = check_boundary_cond_tf(h, self.g, self.G, 100)

        return 0.5 * (main_loss + boundary_loss)
        
