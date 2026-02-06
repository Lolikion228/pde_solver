from env import *
import matplotlib.pyplot as plt
import keras
from keras._tf_keras.keras.layers import *
from env import *
import tensorflow as tf



def train_step(model, opt, problem):
    with tf.GradientTape() as tape:
        loss = problem.compute_loss(model)
    gradients = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(model.trainable_weights, gradients))
    return loss



def test1():

    model = keras.Sequential([
        Input(shape=(2,)),
        Dense(16, activation="tanh"),
        Dense(64, activation="tanh"),
        Dense(16, activation="tanh"),
        Dense(1)
    ])

    opt = keras.optimizers.SGD(1e-3)
    

    def I(x):
        I1 = tf.logical_and(0 <= x[:,0], x[:,0] <= 1)
        I2 = tf.logical_and(0 <= x[:,1], x[:,1] <= 1)
        R = tf.logical_and(I1, I2)
        res = tf.cast(R, tf.float64)
        return res


    # def _gamma1(t, side_length=1.0):
    #     if 0 <= t < 1: 
    #         x = t * side_length
    #         y = 0.0
    #     elif 1 <= t < 2: 
    #         x = side_length
    #         y = (t - 1) * side_length
    #     elif 2 <= t < 3:  
    #         x = side_length - (t - 2) * side_length
    #         y = side_length
    #     else: 
    #         x = 0.0
    #         y = side_length - (t - 3) * side_length
        
    #     return np.array([x, y])

    def gamma1(t, side_length=1.0):
        cond1 = tf.logical_and(0<=t, t<1)
        x1 = t * side_length
        y1 = tf.zeros(t.shape, dtype=tf.float64)

        cond2 = tf.logical_and(1<=t, t<2)
        x2 = tf.ones(t.shape, dtype=tf.float64) * side_length
        y2 = (t - 1) * side_length

        cond3 = tf.logical_and(2<=t, t<3)
        x3 = side_length - (t - 2) * side_length
        y3 = side_length

        #cond4 = tf.logical_and(3<=t, t<=4)
        x4 = tf.zeros(t.shape, dtype=tf.float64)
        y4 = side_length - (t - 3) * side_length
        

        x = tf.where(cond1, x1,
                tf.where(cond2, x2,
                        tf.where(cond3, x3, x4)))
        y = tf.where(cond1, y1,
                    tf.where(cond2, y2,
                            tf.where(cond3, y3, y4)))
    
        res = tf.stack([x, y], axis=-1)

        return res
        
    
    # def _g(x):
    #     if x[0]==0:
    #         return 0
    #     if x[0]==1:
    #         return 0
    #     if x[1]==0:
    #         return 0
    #     if x[1]==1:
    #         return 3 * np.sin(3*np.pi*x[0])

    def g(x):
        res = tf.where(tf.logical_or(x[:,0] * x[:,1]==0, x[:,0]==1),
                       0.0, 3. * tf.sin(3 * np.pi * x[:,0]) )
        return res
    
    G1 = Domain(
        np.array([0.,0.]),
        np.array([0.,0.]),
        I,
        gamma1,
        a=0,
        b=4
    )

    def c1(x):
        return tf.constant(1., dtype=tf.float64)
    
    def c2(x):
        return tf.constant(1., dtype=tf.float64)
    
    coefs = {"xx": c1, "yy":c2}

    P1 = Problem(G1, g, coefs)

    # def h1(x):
    #     if x.ndim==1:
    #         return 3 / np.sinh(4 * np.pi) * np.sin(4 * np.pi * x[0]) * np.sinh(7 * np.pi * x[1])
    #     elif x.ndim==2:
    #         d = x.shape[0]
    #         y = np.zeros(d)
    #         for i in range(d):
    #             y[i] = 3 / np.sinh(3 * np.pi) * np.sin(3 * np.pi * x[i][0]) * np.sinh(3 * np.pi * x[i][1])
    #         return y
    
    def h1(x):
        # print("assa",x)
        if x.shape.rank==1:
            res = tf.constant(3, dtype=tf.float64) \
                  / tf.sinh(tf.cast(3. * np.pi,dtype=tf.float64)) \
                  * tf.sin(3 * np.pi * x[0]) \
                  * tf.sinh(3 * np.pi * x[1])
        if x.shape.rank==2:
            res = tf.constant(3, dtype=tf.float64) \
                  / tf.sinh(tf.cast(3. * np.pi,dtype=tf.float64)) \
                  * tf.sin(3 * np.pi * x[:,0]) \
                  * tf.sinh(3 * np.pi * x[:,1])
        return res

    # def h3(x):
    #     if x.ndim==1:
    #         return 4.01 / np.sinh(6 * np.pi) * np.sin(5 * np.pi * x[0]) * np.sinh(4 * np.pi * x[1])
    #     elif x.ndim==2:
    #         d = x.shape[0]
    #         y = np.zeros(d)
    #         for i in range(d):
    #             y[i] = 5.01 / np.sinh(16 * np.pi) * np.sin(10 * np.pi * x[i][0]) * np.sinh(4 * np.pi * x[i][1])
    #         return y
    
    def h3(x):
        if x.ndim==1:
            res = 5 / tf.sinh(3 * np.pi) * tf.sin(10 * np.pi * x[0]) \
                * tf.sinh(3 * np.pi * x[1])
        if x.ndim==2:
            res = 5 / tf.sinh(3 * np.pi) * tf.sin(10 * np.pi * x[:,0]) \
                * tf.sinh(3 * np.pi * x[:,1])
        return res
        
        
    # def h2(x):
    #     if x.ndim==1:
    #         return 2 * x[0] + x[1]
    #     elif x.ndim==2:
    #         d = x.shape[0]
    #         y = np.zeros(d)
    #         for i in range(d):
    #             y[i] = 2*x[i][0] + x[i][1]
    #         return y
    
    def h2(x):
        if x.shape.rank==1:
            res = 2. * x[0] + x[1]
        if x.shape.rank==2:
            res = 2. * x[:,0] + x[:,1]
        return res
        
    print(P1.compute_loss(h1))
    print(P1.compute_loss(h2))
    # print(P1.compute_loss(h3))

    # x = np.linspace(0, 1, 100)
    # y = np.linspace(0, 1, 100)
    # X, Y = np.meshgrid(x,y)
    # z = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    # Z1 = h1(tf.constant(z, dtype=tf.float64)).numpy().reshape((100,100))
    # Z2 = h2(tf.constant(z, dtype=tf.float64)).numpy().reshape((100,100))


    # for i in range(8):
    #     l = train_step(model, opt, P1)
    #     print(f"epoch: {i}  ||  loss: {l}")

    # plt.figure(figsize=(20, 8))

    # plt.subplot(1, 2, 1)
    # plt.title("exact solution")
    # contour = plt.contourf(X, Y, Z1, 20, cmap='plasma')
    # plt.colorbar(contour)
    # plt.xlabel('x')
    # plt.ylabel('y')

    # plt.subplot(1, 2, 2)
    # plt.title("random function")
    # contour = plt.contourf(X, Y, Z2, 20, cmap='plasma')
    # plt.colorbar(contour)
    # plt.xlabel('x')
    # plt.ylabel('y')

    # plt.show()
    # plt.close()


test1()

