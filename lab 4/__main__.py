import numpy as np
import random
import matplotlib.pyplot as plot
from numpy import linalg as LA, sqrt
import scipy.stats
import pandas as pd
from scipy.stats import t as Student
import csv
import math

Fs = Student.ppf(1 - 0.05/2, 96) 

def u(x1, x2, teta):
    x1 = np.array(x1)
    x2 = np.array(x2)
    f = [1, 1 / x1, x2**2, x1*x2]
    nu = np.dot(teta, f)
    return nu

def calc_e(u):
    w = calc_w(u) 
    sigma = np.sqrt(0.1 * w) 
    return np.random.normal(0, sigma)

def calc_w(u):
    m = calc_m(u)
    u_sr = 0
    for res in u:
        u_sr += (res - m)**2
    return u_sr/(len(u) - 1)

def calc_m(u):
    sum = np.sum(u)
    return sum/len(u)

def calc_y(u):
    e = np.zeros(len(u))
    k = u.shape[0]
    for i in range(k):
        e[i] = calc_e(u)
    
    y = []
    for i in range(k):
        y.append(u[i]+e[i])
    return y, e

def create_x(x1, x2, n):
    cx = np.zeros((len(x1), n))
    cx[:, 0] = 1
    cx[:, 1] = 1/x1
    cx[:, 2] = x2**2
    cx[:, 3] = x1 * x2
    return cx

def create_z(d):
    Z = np.array([0,0])
    for i in range(len(d)):
        newrow = [1, d[i]]    
        Z = np.vstack([Z, newrow])
    Z = np.delete(Z, 0, axis = 0) 
    return Z

def mnk(cx, y):
    cy = np.copy(y)
    teta = (np.dot(cx.transpose(), cx))
    teta = LA.inv(teta)
    teta = np.dot(teta, cx.transpose())
    teta = np.dot(teta, cy)
    return teta

def obob_mnk(cx, V, y):
    cy = np.copy(y)

    teta = (np.dot(cx.transpose(), LA.inv(V)))
    teta = np.dot(teta, cx)
    teta = LA.inv(teta)
    teta = np.dot(teta, cx.transpose())
    teta = np.dot(teta, LA.inv(V))
    teta = np.dot(teta, cy)
    return teta


def calc_RSS(x, y, teta):
    rss = np.dot(np.transpose(y - np.dot(x, teta)), y - np.dot(x, teta))
    return rss

def calc_RSSH(x, y, teta):
    rssh = 0
    for i in range(len(y)): 
        rssh += (y[i] - np.average(y))**2
    return rssh

def create_table(df):
    fig, ax = plot.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')   
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()


def calc_d_vozm(x1, x2):
    d = []
    for i in range (len(x1)):
        d.append(math.sqrt((x1[i]-1) ** 2 + x2[i] ** 2))
    return d

def calc_V(disp, n):
    V = np.zeros((n, n))
    for i in range(n):
        V[i][i] = disp[i]    
    return V

def test_breusha_pagana(c, resZ, Z):
    resZ = np.delete(resZ,0,0) # удаляем свободный член
    Z = np.delete(Z,0,1) # здесь также удаляем свободный член
    ESS = 0
    avg = calc_m(c)
    cc = np.dot(resZ, np.transpose(Z))
    for i in range(len(c)):
        ESS += (cc[i] - avg)**2
    print(ESS/2," ? ",  3.84)
    return cc

def mnk_z(Z, c):
    # Оценивание параметров alpha через обыкновенный МНК (XtX)-1*Xt*Y
    Zt = np.transpose(Z)
    skobka = np.dot(Zt,Z)
    skobka = np.linalg.matrix_power(skobka, -1)
    skobka = np.dot(skobka,Zt)
    resZ = np.dot(skobka, c)
    return resZ

def u_ed(x1, x2, teta):
    return teta[0]*1 + teta[1]* (1 / x1) + teta[2]*x2**2 + teta[3]* x1* x2

def test_goldfelda(X, d, N, u, teta):
    #    Тест №2 - Голдфельда-Квандтона
    #    Упорядочим матрицу наблюдений по величине модуля суммы двух факторов

    XwDisp = np.array(X)
    d = np.reshape(d,(N,1))
    XwDisp = np.append(XwDisp, d, axis=1)
    u = np.reshape(u,(N,1))
    XwDisp = np.append(XwDisp, u, axis=1) 

    Xsorted = np.array(sorted(XwDisp, key=lambda a: abs(u_ed(a[1], a[2], teta))))
    nc = int(N/3)
    X1 = Xsorted[:nc]
    X2 = Xsorted[2*nc:]

    tetha = mnk(X1[:,:4], X1[:,5])
    e1 = X1[:,4] - np.dot(X1[:,:4],tetha)

    tetha = mnk(X2[:,:4], X2[:,5])
    e2 = X2[:,4] - np.dot(X2[:,:4],tetha)

    print("RSS2/RSS1 = ",(np.dot(np.transpose(e2), e2))/(np.dot(np.transpose(e1), e1)))

def main():
    teta = [0.3, 0.14, 0.432, 3]
    n = 300
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for i in range(n):
        x1[i] = random.uniform(0, 2)
        x2[i] = random.uniform(-1, 1)
    U = np.zeros(n)
    U = u(x1, x2, teta)
    y, e = calc_y(U)
    M = calc_m(U)
    d = calc_d_vozm(x1, x2)
    cx = create_x(x1, x2, len(teta))
    Z = create_z(d)

    mnk_teta = mnk(cx, y)
    e_oz = y - np.dot(cx, mnk_teta)
    sigmaTheoretical = (np.dot(np.transpose(e_oz), e_oz)) / n
    sigma = (e_oz * e_oz)/sigmaTheoretical

    # обычный мнк c Z
    resZ = mnk_z(Z, sigma)
    # test 1
    sigma_oz = test_breusha_pagana(sigma, resZ, Z)

    # обобщенный мнк
    V = calc_V(sigma_oz, n)
    obob_teta = obob_mnk(cx, V, y)

    # test 2
    res = test_goldfelda(cx, d, n, U, teta)
    print(Fs)
    df = pd.DataFrame()
    df['Тетта задаваемое'] = teta
    df['Тетта по  МНК'] = mnk_teta
    df['Тетта по обобщенному МНК'] = obob_teta
    create_table(df)
    d2 = 0
    for i in range(len(obob_teta)):
        d2 += (obob_teta[i] - teta[i])**2
    print("Расстояние обобщенного МНК - ", d2)

    d2_m = 0
    for i in range(len(mnk_teta)):
        d2_m += (mnk_teta[i] - teta[i])**2
    print("Расстояние МНК - ", d2_m)
    plot.show()

main()