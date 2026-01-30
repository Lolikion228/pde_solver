from integration import *
from diff import *
import tensorflow as tf


eps = 1e-4
N = 10000

class Problem:

    def __init__(self, G, g, coefs):
        self.G = G
        self.g = g 
        self.coefs = coefs
        
    
    def compute_L(self, f, x):
        L = 0
        
        if self.coefs[0] != None: 
            L += self.coefs[0](x) * pddf(f, x, 0, eps)
        if self.coefs[1] != None: 
            L += self.coefs[1](x) * pddf(f, x, 1, eps)
        if self.coefs[2] != None: 
            L += self.coefs[2](x) * pdf(f, x, 0, eps)
        if self.coefs[3] != None: 
            L += self.coefs[3](x) * pdf(f, x, 1, eps)
        if self.coefs[4] != None: 
            L += self.coefs[4](x) * f(x)

        return L

    
    def compute_loss(self, h):
        main_loss = mc_int(self.G, lambda x: self.compute_L(h,x)**2 , N)
        boundary_loss = check_boundary_cond(h, self.g, self.G, N)
        return 0.5 * ( main_loss + boundary_loss )
        
