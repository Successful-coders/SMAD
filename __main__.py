import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA

def u(x1, x2, teta):
    f = np.zeros(4)
    f = [1, 1/x1, x2 ** 2, x1*x2]
    nu = np.dot(teta, f)
    return nu


def calc_e(u):
    w = calc_w(u) 
    sigma = np.sqrt(0.1 * w) 
    return round(np.random.normal(0, sigma), 5)

def calc_w(u):
    m = calc_m(u, len(u))
    u_sr = 0
    for res in u:
        u_sr += (res - m) ** 2
    return u_sr/(len(u) - 1)

def calc_m(u, n):
    sum = np.sum(u)
    return sum/n

def calc_y(u):
    e = np.zeros(len(u))
    k = u.shape[0]
    for i in range(k):
        e[i] = calc_e(u)
    
    y = []
    for i in range(k):
        y.append(u[i]+e[i])
    
    return y, e

def create_plot(x,y, u):
    fig, ax = plt.subplots()
    ax.set_title("СМАД")
    ax.plot(x,y, label='Зашумленные данные')
    ax.plot(x,u, label='Незашумленные данные')
    ax.set_ylabel("Время дожития, (лет)")
    ax.set_xlabel("Номер пациента")
    ax.legend()
    ax.grid()

    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    plt.show()

def mnk(x1, x2, y):
    cx = np.zeros((len(x1), 4))
    cx[:, 0] = 1
    cx[:, 1] = 1/x1
    cx[:, 2] = x2 **2
    cx[:, 3] = x1 * x2
    cy = np.copy(y)

    teta = (np.dot(cx.transpose(), cx))
    teta = LA.inv(teta)
    teta = np.dot(teta, cx.transpose())
    teta = np.dot(teta, cy)

    return teta, cx

    


def __main__():
    teta = [0.3, 0.12, 0.432, 3]
    n = 40
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x = np.zeros(n)
    zero = np.zeros(n)
    x[0] = -1
    for i in range(40):
        x1[i] = random.uniform(-1, 1)
        x2[i] = random.uniform(-1, 1)
    for i in range(39):
        x[i+1] = x[i] +1/40
    U = np.zeros(40)
    U = u(x1, x2, teta)
    y, e = calc_y(U)
    m, cx = mnk(x1, x2, y)

    e_oz = y - np.dot(cx, m)
    sigma_kv = np.dot(e_oz.transpose(), e_oz)/ n - len(teta)
    sigma = np.dot(e.transpose(),e) / n - len(teta)
    F = sigma_kv/sigma

    create_plot(x, y, U)


__main__()