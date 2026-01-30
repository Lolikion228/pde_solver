from integration import *


def int_test1():
    x1 = np.array([-13,30,220])
    x2 = np.array([-11,40,200])
    x = sample(x1, x2, 3)
    print(x)

def int_test2():
    def f(x):
        if x.ndim==1:
            return 12 * x[0]**2 * x[1]**3
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = 12 * x[i][0]**2 * x[i][1]**3
            return y
        else: 
            raise Exception("aaa")
        
    def I(x):
        if x.ndim==1:
            return (0<=x[0]) * (x[0] <= 1) * (0<=x[1]) * (x[1] <= 1)
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = (0<=x[i][0]) * (x[i][0] <= 1) * (0<=x[i][1]) * (x[i][1] <= 1)
            return y
        else:
            raise Exception("aaaa")

    x1 = np.array([-1,-1])
    x2 = np.array([2,2])

    G1 = Domain(x1, x2, I, None)

    print(mc_int(G1, f, 10000))


def int_test3():
    def f(x):
        if x.ndim==1:
            return (x[0]**2 + x[1]**2) / np.pi
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = (x[i][0]**2 + x[i][1]**2) / np.pi
            return y
        else: 
            raise Exception("aaa")
        
    def I(x):
        if x.ndim==1:
            return (x[0]**2 + x[1]**2 <= 4) * (x[1]>=0)
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = (x[i][0]**2 + x[i][1]**2 <= 4) * (x[i][1]>=0)
            return y
        else:
            raise Exception("aaaa")

    x1 = np.array([-3,-3])
    x2 = np.array([3,3])

    G1 = Domain(x1, x2, I, None, 1, 1)

    print(mc_int(G1, f, 10000))



def int_test4():
    a = 0
    b = 3

    def gamma(x):
        if x.ndim==0:
            return np.array([x,0]) 
        elif x.ndim==1:
            d = x.shape[0]
            y = np.zeros(shape=(d,2))
            for i in range(d):
                y[i] = np.array([x[i],0]) 
            return y
        else:
            raise Exception("aaaa")
    


    G1 = Domain(None, None, None, gamma, a, b)

    def f1(x):
        if x.ndim==1:
            return 2*x[0] + 3*x[1]
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = 2*x[i][0] + 3*x[i][1]
            return y
        else: 
            raise Exception("aaa")

    def f2(x):
        if x.ndim==1:
            return 5*x[0] + 2*x[1]
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = 5*x[i][0] + 2*x[i][1]
            return y
        else: 
            raise Exception("aaa")
        
    print(check_boundary_cond(f1, f2, G1, 10000))


def int_test5():
    a = -np.pi / 2
    b = np.pi / 2

    def gamma(x):
        if x.ndim==0:
            return np.array([np.cos(x),np.sin(x)]) 
        elif x.ndim==1:
            d = x.shape[0]
            y = np.zeros(shape=(d,2))
            for i in range(d):
                y[i] = np.array([np.cos(x[i]),np.sin(x[i])]) 
            return y
        else:
            raise Exception("aaaa")
    


    G1 = Domain(None, None, None, gamma, a, b)

    def f1(x):
        if x.ndim==1:
            return x[0]**2.05 + x[1]**2
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = x[i][0]**2.05 + x[i][1]**2
            return y
        else: 
            raise Exception("aaa")

    def f2(x):
        if x.ndim==1:
            return x[0]**2.1 + abs(x[1])**2.2
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = x[i][0]**2.1 + abs(x[i][1])**2.2
            return y
        else: 
            raise Exception("aaa")
        
    print(check_boundary_cond(f1, f2, G1, 1000))


int_test5()