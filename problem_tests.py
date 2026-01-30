from env import *
import matplotlib.pyplot as plt


def test1():
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


    def _gamma1(t, side_length=1.0):
        if 0 <= t < 1: 
            x = t * side_length
            y = 0.0
        elif 1 <= t < 2: 
            x = side_length
            y = (t - 1) * side_length
        elif 2 <= t < 3:  
            x = side_length - (t - 2) * side_length
            y = side_length
        else: 
            x = 0.0
            y = side_length - (t - 3) * side_length
        
        return np.array([x, y])

    def gamma1(x):
        if x.ndim==0:
            return _gamma1(x)
        elif x.ndim==1:
            d = x.shape[0]
            y = np.zeros(shape=(d,2))
            for i in range(d):
                y[i] = _gamma1(x[i])
            return y

    def _g(x):
        if x[0]==0:
            return 0
        if x[0]==1:
            return 0
        if x[1]==0:
            return 0
        if x[1]==1:
            return 3 * np.sin(3*np.pi*x[0])

    def g(x):
        if x.ndim==1:
            return _g(x)
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = _g(x[i])
            return y

    G1 = Domain(
        np.array([0.,0.]),
        np.array([0.,0.]),
        I,
        gamma1,
        0,
        4
    )


    coefs = [lambda x: 1., lambda x: 1., None, None, None]

    P1 = Problem(G1, g, coefs)

    def h1(x):
        if x.ndim==1:
            return 3 / np.sinh(4 * np.pi) * np.sin(4 * np.pi * x[0]) * np.sinh(7 * np.pi * x[1])
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = 3 / np.sinh(3 * np.pi) * np.sin(3 * np.pi * x[i][0]) * np.sinh(3 * np.pi * x[i][1])
            return y
        
    def h3(x):
        if x.ndim==1:
            return 4.01 / np.sinh(6 * np.pi) * np.sin(5 * np.pi * x[0]) * np.sinh(4 * np.pi * x[1])
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = 5.01 / np.sinh(16 * np.pi) * np.sin(10 * np.pi * x[i][0]) * np.sinh(4 * np.pi * x[i][1])
            return y
        
        
    def h2(x):
        if x.ndim==1:
            return 2 * x[0] + x[1]
        elif x.ndim==2:
            d = x.shape[0]
            y = np.zeros(d)
            for i in range(d):
                y[i] = 2*x[i][0] + x[i][1]
            return y
        
    print(P1.compute_loss(h1))
    print(P1.compute_loss(h2))
    print(P1.compute_loss(h3))

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x,y)
    z = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    Z1 = h1(z).reshape((100,100))
    Z2 = h2(z).reshape((100,100))


    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.title("exact solution")
    contour = plt.contourf(X, Y, Z1, 20, cmap='plasma')
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.title("random function")
    contour = plt.contourf(X, Y, Z2, 20, cmap='plasma')
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
    plt.close()


test1()