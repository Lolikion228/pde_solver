from integration import *


def int_test1():
    x1 = np.array([-13,30,220])
    x2 = np.array([-11,40,200])
    x = sample(x1, x2, 3)
    print(x)

# ANS = 1.0
def int_test2():
    
    def f(x):
        res = 12 * x[:,0]**2 * x[:,1]**3
        return res
    
        
    def I(x):
        I1 = tf.logical_and(0 <= x[:,0], x[:,0] <= 1)
        I2 = tf.logical_and(0 <= x[:,1], x[:,1] <= 1)
        R = tf.logical_and(I1, I2)
        res = tf.cast(R, tf.float32)
        return res

    x1 = np.array([-1., -1])
    x2 = np.array([2,    2.])

    G1 = Domain(x1, x2, I, None, 1, 1)

    print(mc_int(G1, f, 100000))

# ANS = 4.0
def int_test3():
    def f(x):
        res = (x[:,0]**2 + x[:,1]**2) / np.pi
        return res
        
    def I(x):
        y = tf.logical_and(x[:,0]**2 + x[:,1]**2 <= 4, x[:,1]>=0)
        res = tf.cast(y, dtype=tf.float32)
        return res


    x1 = np.array([-3., -3])
    x2 = np.array([3.,   3])

    G1 = Domain(x1, x2, I, None, 1, 1)

    print(mc_int(G1, f, 100000))



def int_test4():
    a = 0
    b = 3

    def gamma(x):
        res = tf.stack([x, tf.zeros(x.shape[0], dtype=tf.float32)], axis=1)
        return res
     
    G1 = Domain(None, None, None, gamma, a, b)

    def f1(x):
        res = 2*x[:,0] + 33*x[:,1]
        return res
 

    def f2(x):
        res = 2.2*x[:,0] + 22*x[:,1]
        return res
        
    print(check_boundary_cond(f1, f2, G1, 10000))


def int_test5():
    a = -np.pi / 2
    b = np.pi / 2

    def gamma(x):
        res = tf.stack([tf.cos(x), tf.sin(x)], axis=1) 
        return res
    
    G1 = Domain(None, None, None, gamma, a, b)

    def f1(x):
        res = tf.abs(x[:,0])**3.2 + tf.abs(x[:,1])**2.1
        return res
       

    def f2(x):
        res = tf.abs(x[:,0])**3.1 + tf.abs(x[:,1])**2.0
        return res
        
    print(check_boundary_cond(f1, f2, G1, 10000))


int_test5()