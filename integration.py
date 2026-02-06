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
    """
    TensorFlow-совместимая версия интеграции с квадратурой Гаусса.
    Все операции выполняются в TensorFlow для сохранения графа вычислений.
    """
    # Определяем границы прямоугольника
    x_min = tf.cast(tf.minimum(G.x1[0], G.x2[0]),dtype=tf.float32)
    x_max = tf.cast(tf.maximum(G.x1[0], G.x2[0]),dtype=tf.float32)
    y_min = tf.cast(tf.minimum(G.x1[1], G.x2[1]),dtype=tf.float32)
    y_max = tf.cast(tf.maximum(G.x1[1], G.x2[1]),dtype=tf.float32)
    
    # Получаем точки и веса квадратуры Гаусса через TensorFlow
    # Используем tf.linalg.eigh для получения точек Лежандра
    n = n_points
    
    # Создаем матрицу Якоби для полиномов Лежандра
    i = tf.range(1, n, dtype=tf.float32)
    beta = i / tf.sqrt(4 * i**2 - 1)
    
    # Создаем трехдиагональную матрицу
    diag = tf.zeros((n,), dtype=tf.float32)
    subdiag = beta
    
    # Находим собственные значения и векторы (точки и веса квадратуры Гаусса)
    # Используем tf.linalg.eigh для симметричной трехдиагональной матрицы
    matrix = tf.linalg.diag(diag) + tf.linalg.diag(subdiag, k=1) + tf.linalg.diag(subdiag, k=-1)
    
    # Вычисляем собственные значения и векторы
    eigenvalues, eigenvectors = tf.linalg.eigh(matrix)
    
    # Точки квадратуры - собственные значения
    points_1d = eigenvalues
    
    # Веса квадратуры - квадраты первого элемента собственных векторов
    weights_1d = 2.0 * eigenvectors[0, :]**2
    
    # Преобразуем точки из [-1, 1] в [x_min, x_max] и [y_min, y_max]
    x_points = (points_1d + 1.0) / 2.0 * (x_max - x_min) + x_min
    y_points = (points_1d + 1.0) / 2.0 * (y_max - y_min) + y_min
    
    # Создаем сетку точек в TensorFlow
    xx, yy = tf.meshgrid(x_points, y_points, indexing='ij')
    grid_points = tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], axis=1)
    
    # Создаем сетку весов
    wx, wy = tf.meshgrid(weights_1d, weights_1d, indexing='ij')
    weights = tf.reshape(wx * wy, [-1])
    
    # Масштабируем веса на размеры прямоугольника
    scale_factor = (x_max - x_min) * (y_max - y_min) / 4.0
    scaled_weights = weights * scale_factor
    
    # Вычисляем значения функции
    f_values = f(grid_points)  # [n_points^2]
    
    # Применяем индикаторную функцию
    indicator_values = G.indicator(grid_points)  # [n_points^2]
    
    # Вычисляем интеграл
    integral = tf.reduce_sum(f_values * indicator_values * scaled_weights)
    
    return integral


def check_boundary_cond(f1, f2, G, n):
    x = tf.random.uniform(shape=(n,),
                          minval=G.a, maxval=G.b, dtype=tf.float32)
    path = G.gamma(x)
    delta = f1(path) - f2(path)
    return (G.b - G.a) * tf.reduce_mean(delta**2)

def check_boundary_cond_tf(f1, f2, G, n_points=10):
    """
    Полностью TensorFlow-совместимая версия.
    """
    # Предварительно вычисленные точки и веса Гаусса
    gauss_data = {
        5: {
            'points': tf.constant([-0.9061798459, -0.5384693101, 0.0, 
                                   0.5384693101, 0.9061798459], dtype=tf.float32),
            'weights': tf.constant([0.2369268850, 0.4786286705, 0.5688888889, 
                                    0.4786286705, 0.2369268850], dtype=tf.float32)
        },
        10: {
            'points': tf.constant([-0.9739065285, -0.8650633667, -0.6794095683,
                                   -0.4333953941, -0.1488743390, 0.1488743390,
                                   0.4333953941, 0.6794095683, 0.8650633667,
                                   0.9739065285], dtype=tf.float32),
            'weights': tf.constant([0.0666713443, 0.1494513492, 0.2190863625,
                                    0.2692667193, 0.2955242247, 0.2955242247,
                                    0.2692667193, 0.2190863625, 0.1494513492,
                                    0.0666713443], dtype=tf.float32)
        }
    }
    
    # Выбираем ближайшее доступное количество точек
    if n_points not in gauss_data:
        n_points = min(gauss_data.keys(), key=lambda x: abs(x - n_points))
    
    points_1d = gauss_data[n_points]['points']
    weights_1d = gauss_data[n_points]['weights']
    
    # Преобразуем точки из [-1, 1] в [G.a, G.b]
    t_points = (points_1d + 1.0) / 2.0 * (G.b - G.a) + G.a
    
    # Масштабируем веса
    scaled_weights = weights_1d * (G.b - G.a) / 2.0
    
    # Вычисляем точки на границе
    boundary_points = G.gamma(t_points)  # shape: (n_points, 2)
    
    # Вычисляем значения функций
    f1_values = f1(boundary_points)
    f2_values = f2(boundary_points)
    
    # Разность в квадрате
    diff_squared = (f1_values - f2_values)**2
    
    # Интеграл
    integral = tf.reduce_sum(diff_squared * scaled_weights)
    
    return integral