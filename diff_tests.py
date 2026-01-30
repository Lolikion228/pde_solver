from diff import *
import numpy as np

def test_pipline1(f1,  df1,  ddf1,  a,  b,  n):
    h = 0.1
    
    for i in range(6):
        x = a
        print("h =", h)

        print("x       ", end="")
        print("df_delta   ",end="")
        print("ddf_delta")

        for j in range(n):
            x += (b - a) / n
            df_delta = abs(df1(x) - df(f1, x, h))
            ddf_delta = abs(ddf1(x) - ddf(f1, x, h))
            print(f"{x:>{6}.{4}f} {df_delta:>{8}.{4}e} {ddf_delta:>{8}.{4}e}")
        
        h /= 10
        print()
    


def diff_test1():

    f1   = lambda x: 5 * x**7 + 4 * x**3 + x
    df1  = lambda x: 35 * x**6 + 12 * x**2 + 1
    ddf1 = lambda x: 210 * x**5 + 24 * x**1

    a = -2
    b = 2
    n = 6

    test_pipline1(f1, df1, ddf1, a, b, n)



def diff_test2():
    f1   = lambda x: np.sin(x) + np.exp(2 * x) + np.cos(x) * np.cos(x)
    df1  = lambda x: np.cos(x) + 2 * np.exp(2 * x) - 2 * np.cos(x) * np.sin(x)
    ddf1 = lambda x: -np.sin(x) + 4 * np.exp(2 * x) - 2* np.cos(2*x)


    a = -5
    b = 5
    n = 6

    test_pipline1(f1, df1, ddf1, a, b, n)



