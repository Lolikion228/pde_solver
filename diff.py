import tensorflow as tf

def diff(f, x):
    x = tf.Variable(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = f(x)
    return tape.gradient(y,x)

def diff2(f, x):
    x = tf.Variable(x, dtype=tf.float32)
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            y = f(x)
        dy = tape1.gradient(y,x)
    J = tape2.jacobian(dy,x)
    ddy = tf.linalg.diag_part(J)
    return ddy


