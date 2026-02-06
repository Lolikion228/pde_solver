from env import *
import matplotlib.pyplot as plt
import keras
from keras._tf_keras.keras.layers import *
from env import *
import tensorflow as tf

class FlexibleSequential(keras.Sequential):
    def call(self, inputs):
        was_1d = inputs.shape.rank == 1
        
        if was_1d:
            inputs = tf.expand_dims(inputs, axis=0)
        
        outputs = super().call(inputs)

        if was_1d:
            outputs = tf.squeeze(outputs, axis=0)
        
        return outputs

def train_step(model, opt, problem):
    with tf.GradientTape() as tape:
        loss = problem.compute_loss(model)
    gradients = tape.gradient(loss, model.trainable_weights)
    grads_and_vars = zip(gradients, model.trainable_weights)
    opt.apply_gradients(grads_and_vars)
    return loss


def test1():

    model = FlexibleSequential([
        Input(shape=(2,)),
        Dense(16, activation="tanh"),
        Dense(64, activation="tanh"),
        Dense(256, activation="tanh"),
        Dense(64, activation="tanh"),
        Dense(16, activation="tanh"),
        Dense(1, dtype=tf.float64)
    ])
    
    opt = keras.optimizers.SGD(1e-4)
    

    def I(x):
        I1 = tf.logical_and(0 <= x[:,0], x[:,0] <= 1)
        I2 = tf.logical_and(0 <= x[:,1], x[:,1] <= 1)
        R = tf.logical_and(I1, I2)
        res = tf.cast(R, tf.float64)
        return res
    

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
        

    def g(x):
        res = tf.where(tf.logical_or(x[:,0] * x[:,1]==0, x[:,0]==1),
                       0.0, 3. * tf.sin(3 * np.pi * x[:,0]) )
        return res
    
    G1 = Domain(
        np.array([-1.,-1.]),
        np.array([1.,1.]),
        I,
        gamma1,
        a=0,
        b=4
    )

    def c1(x):
        return tf.ones(shape=x.shape[0], dtype=tf.float64)
    
    def c2(x):
        return tf.ones(shape=x.shape[0], dtype=tf.float64)
    
    coefs = {"xx": c1, "yy":c2}

    P1 = Problem(G1, g, coefs)
    
    def h1(x):
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

    def h2(x):
        if x.shape.rank==1:
            res = 2. * x[0]**2 + x[1]**2
        if x.shape.rank==2:
            res = 2. * x[:,0]**2 + x[:,1]**2
        return res
    
    def h3(x):
        if x.shape.rank==1:
            res = tf.constant(2, dtype=tf.float64) \
                  / tf.sinh(tf.cast(6. * np.pi,dtype=tf.float64)) \
                  * tf.sin(4 * np.pi * x[0]) \
                  * tf.sinh(6 * np.pi * x[1])
        if x.shape.rank==2:
            res = tf.constant(2, dtype=tf.float64) \
                  / tf.sinh(tf.cast(6. * np.pi,dtype=tf.float64)) \
                  * tf.sin(4 * np.pi * x[:,0]) \
                  * tf.sinh(6 * np.pi * x[:,1])
        return res
        

    # print("here")
    # print(P1.compute_loss(h1))
    # print(P1.compute_loss(h2))
    # print(P1.compute_loss(h3))

    # x = np.linspace(0, 1, 100)
    # y = np.linspace(0, 1, 100)
    # X, Y = np.meshgrid(x,y)
    # z = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    # Z1 = h1(tf.constant(z, dtype=tf.float64)).numpy().reshape((100,100))
    # Z2 = h2(tf.constant(z, dtype=tf.float64)).numpy().reshape((100,100))


    for i in range(16):
        l = train_step(model, opt, P1)
        print(f"epoch: {i}  ||  loss: {l}")

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

