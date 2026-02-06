import tensorflow as tf

def diff(f, x):
    x = tf.constant(x, dtype=tf.float32)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = f(x)
    return tape.gradient(y,x)

# def diff2(f, x):
#     x = tf.Variable(x, dtype=tf.float32)
#     with tf.GradientTape() as tape2:
#         with tf.GradientTape() as tape1:
#             y = f(x)
#         dy = tape1.gradient(y,x)
#     J = tape2.jacobian(dy,x)
#     if J is None:
#         return tf.zeros(shape=dy.shape, dtype=tf.float32)
#     else:
#         ddy = tf.linalg.diag_part(J)
#     return ddy


def diff2(f, x):
    x_var = tf.Variable(x, dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x_var)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_var)
            y = f(x_var)
        dy = tape1.gradient(y, x_var)
    
    
    if x.shape.rank==1:
        J = tape2.jacobian(dy, x_var)
    if x.shape.rank==2:
        J = tape2.batch_jacobian(dy, x_var)  # shape: (batch_size, 2, 2)
    
    del tape2
    if J is None:
        return tf.zeros_like(x)
    else:
        return tf.linalg.diag_part(J)  # shape: (batch_size, 2)
    
