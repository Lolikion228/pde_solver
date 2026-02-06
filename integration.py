import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Domain:
    def __init__(self, x1, x2, indicator, gamma, a, b):
        self.x1 = x1 # np.array
        self.x2 = x2 # np.array
        self.indicator = indicator # function defined on 2_dsamples
        self.a = a
        self.b = b
        self.gamma = gamma # v-function defined on 1d_samples


def sample(x1, x2, n):
    d = x1.shape[0]
    U = tf.random.uniform(shape=(n,d), dtype=tf.float32)
    x = U * (x2 - x1) + x1
    return x


def mc_int(G, f, n):
    sides = tf.cast(tf.abs(G.x2 - G.x1),dtype=tf.float32)
    vol = tf.reduce_prod(sides)
    x = sample(G.x1, G.x2, n)
    mask = G.indicator(x)
    vals = f(x) * mask
    mean = tf.reduce_mean(vals)
    res = vol * mean
    return res



def integrate_over_domain(G, f, n_points=1000):
    # Определяем границы прямоугольника
    x_min = min(G.x1[0], G.x2[0])
    x_max = max(G.x1[0], G.x2[0])
    y_min = min(G.x1[1], G.x2[1])
    y_max = max(G.x1[1], G.x2[1])
    
    # Получаем точки и веса квадратуры Гаусса
    points_1d, weights_1d = np.polynomial.legendre.leggauss(n_points)
    
    # Преобразуем точки из [-1, 1] в [x_min, x_max] и [y_min, y_max]
    x_points = (points_1d + 1) / 2 * (x_max - x_min) + x_min
    y_points = (points_1d + 1) / 2 * (y_max - y_min) + y_min
    
    # Создаем сетку точек
    xx, yy = np.meshgrid(x_points, y_points, indexing='ij')
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)  # [n_points^2, 2]
    
    # Создаем сетку весов
    wx, wy = np.meshgrid(weights_1d, weights_1d, indexing='ij')
    weights = (wx * wy).flatten()  # [n_points^2]
    
    # Масштабируем веса на размеры прямоугольника
    scale_factor = (x_max - x_min) * (y_max - y_min) / 4
    scaled_weights = weights * scale_factor
    
    # Преобразуем в тензоры TensorFlow
    grid_points_tf = tf.convert_to_tensor(grid_points, dtype=tf.float32)
    scaled_weights_tf = tf.convert_to_tensor(scaled_weights, dtype=tf.float32)
    
    # Вычисляем значения функции
    f_values = f(grid_points_tf)  # [n_points^2]
    
    # Применяем индикаторную функцию
    indicator_values = G.indicator(grid_points_tf)  # [n_points^2]
    
    # Вычисляем интеграл
    integral = tf.reduce_sum(f_values * indicator_values * scaled_weights_tf)
    
    return integral


def check_boundary_cond(f1, f2, G, n):
    x = tf.random.uniform(shape=(n,),
                          minval=G.a, maxval=G.b, dtype=tf.float32)
    path = G.gamma(x)
    delta = f1(path) - f2(path)
    return (G.b - G.a) * tf.reduce_mean(delta**2)

