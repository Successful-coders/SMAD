import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.stats
from requests_html import HTMLSession

def u(x1, x2, teta):
    f = np.zeros(4)
    f = [1, 1/x1, x2 ** 2, x1*x2]
    nu = np.dot(teta, f)
    return nu

def calc_e(u):
    w = calc_w(u) 
    sigma = np.sqrt(0.1 * w) 
    return np.random.normal(0, sigma)

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
        e[i] = round(calc_e(u), 5)
    
    y = []
    for i in range(k):
        y.append(round(u[i]+e[i], 5))
    return y, e

def create_plot(x, y, u):
    fig, ax = plt.subplots()
    ax.set_title("СМАД")
    # ax.plot(x,y, label='Зашумленные данные')
    ax.plot(x,u, label='Незашумленные данные')
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
    for i in range(4):  
        teta[i] = round(teta[i], 5)
    return teta, cx

def __main__():
    teta = [0.3, 0.14, 0.432, 3]
    n = 40
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x = np.zeros(n)
    zero = np.zeros(n)
    x[0] = -1

    for i in range(40):
        x1[i] = round(random.uniform(0, 2),5)
        x2[i] = round(random.uniform(-1, 1),5)
    # for i in range(39):
        # x[i+1] = x[i] + 2/40
    
    U = np.zeros(40)
    U = u(x1, x2, teta)

    y, e = calc_y(U)
    m, cx = mnk(x1, x2, y)

    e_oz = y - np.dot(cx, m)

    sigma_kv = np.dot(e_oz.transpose(), e_oz)/ (n - len(teta))
    sigma = np.dot(e.transpose(),e) / (n - len(teta))
    F = sigma_kv/sigma

    Ft = scipy.stats.t.ppf(0.95, 36, np.inf)

    y_oz = np.dot(cx, m)
    for i in range(len(y_oz)):
        y_oz[i] = round(y_oz[i], 5)

    f = open('results.csv', 'w')
    res = 'x1\tx2\tu\ty\te\ty^\ty-y^\n'
    f.write(res)
    # create_plot(x, y, U)
    for i in range(n):
        res = f"{x1[i]}\t{x2[i]}\t{U[i]}\t{y[i]}\t{e[i]}\t{y_oz[i]}\t{round(y[i]-y_oz[i], 5)}\n"
        f.write(res)
    f.close()


    # print('teta oz = ', m)
    # print('F = ', F )
    print('Ft = ', Ft)


__main__()