import numpy as np
import random
import matplotlib.pyplot as plot
from numpy import linalg as LA, sqrt
import scipy.stats
from requests_html import HTMLSession
import pandas as pd

def u(x1, x2, teta):
    f = np.zeros(4)
    f = [1, 1/x1, x2 ** 2, x1*x2, x1 / x2]
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
        e[i] = calc_e(u)
    
    y = []
    for i in range(k):
        y.append(u[i]+e[i])
    return y, e

def create_plot(x, y, u):
    fig, ax = plot.subplots()
    ax.set_title("СМАД")
    # ax.plot(x,y, label='Зашумленные данные')
    ax.plot(x, u, label='Незашумленные данные')
    ax.legend()
    ax.grid()

    fig.set_figheight(5)
    fig.set_figwidth(10)
    plot.show()
    plot.show()

def mnk(x1, x2, y, n):
    cx = np.zeros((len(x1), n))
    cx[:, 0] = 1
    cx[:, 1] = 1/x1
    cx[:, 2] = x2 **2
    cx[:, 3] = x1 * x2
    cx[:, 4] = x1 / x2
    cy = np.copy(y)

    teta = (np.dot(cx.transpose(), cx))
    teta = LA.inv(teta)
    teta = np.dot(teta, cx.transpose())
    teta = np.dot(teta, cy)
    return teta, cx

def calc_djj(x):
    matrix = np.linalg.inv(np.dot(x.transpose(), x))
    djj = []
    for i in range(len(matrix)):
        djj.append(matrix[i][i])
    return djj

def create_dov_int(cx, F, teta, sigma_kv, djj):
    D_verh = np.zeros(len(teta))
    D_nizh = np.zeros(len(teta))
    for i in range(len(teta)):
        D_verh[i] = teta[i] - F * np.sqrt(sigma_kv *djj[i])
        D_nizh[i] = teta[i] + F * np.sqrt(sigma_kv *djj[i])
    return D_verh, D_nizh

def calc_neznach_hipot_param(teta, djj, sigma):
    F = []
    for i in range(len(teta)):
        F.append((teta[i]) ** 2 / (sigma * djj[i]))
    return F

def calc_RSS(sigma, n, m):
    rss = (n-m) * sigma
    return rss

def calc_RSSH(x, y, teta):
    rssh = np.dot(np.transpose(y - np.dot(x, teta)), y - np.dot(x, teta))
    return rssh

def calc_zn_hip(rss, rssh, n, m, q):
    F = ((rssh - rss) / q) / (rss / (n - m))
    return F


def create_table(df):
    fig, ax = plot.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')   
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()

def __main__():
    teta = [0.3, 0.14, 0.432, 3, 0.00001]
    n = 40
    x1 = np.zeros(n)
    x2 = np.zeros(n)

    x1 = np.array([ 0.32390023, -0.34725259,  0.78381968, -0.24713599, -0.92583476, -0.76140316,
            0.78438351,  0.06926699, -0.19419177, -0.20446248,  0.36498275,  0.6430685,
            -0.69000552,  0.53620227, -0.36596288,  0.09916682,  0.69863079, -0.77057757,
            0.62452704, -0.38963251, -0.36289536,  0.70176654,  0.76653001, -0.8264534,
            -0.46460932,  0.71063226,  0.23511767, -0.71029019, 0.02145435, 0.38076295,
            0.10906815, -0.68266406,  0.12610703,  0.89168848,  0.79420493, -0.1765881,
            0.24414797,  0.84046955,  0.45996973,  0.8613737 ])
    x2 = np.array([-0.69762794,  0.66021212,  0.65267716, -0.24793652, -0.26103093, -0.02192529,
            0.49963626, -0.82567682, -0.24142269, -0.66004223,  0.05343135,  0.8917568,
            0.68045318,  0.11433146, -0.08409676,  0.02706899, -0.87425862,  0.5083733,
            -0.96566545, -0.38398395, -0.9093806,   0.68809117,  0.74604826, -0.15810338,
            0.14329951,  0.83901296,  0.13880737,  0.81852986, -0.94341511,  0.60822667,
            -0.0312237,  -0.50614177,  0.08375906,  0.85850998,  0.19358785,  0.42399955,
            0.43474397,  0.78668444,  0.93962503,  0.49517424])
    U = np.zeros(40)
    U = u(x1, x2, teta)

    y, e = calc_y(U)
    mnk_teta, cx = mnk(x1, x2, y, 5)
    e_oz = y - np.dot(cx, mnk_teta)
    y_oz = np.dot(cx, mnk_teta)

    sigma_kv = np.dot(e_oz.transpose(), e_oz)/ (n - len(teta))
    sigma = np.dot(e.transpose(),e) / (n - len(teta))
    print ('sigma kv=', sigma_kv)
    print ('sigma=', sigma)
    F = sigma_kv/sigma

    Ft = scipy.stats.t.ppf(0.95, 36)
    djj = calc_djj(cx)
    D_verh, D_nizh = create_dov_int(cx, Ft, mnk_teta, sigma_kv, djj)

    F_zn_param = calc_neznach_hipot_param(mnk_teta, djj, sigma_kv)
    temp = []
    Fnm = scipy.stats.f.ppf(1-0.05, 1, 35)
    for i in range(len(mnk_teta)):
        if(F_zn_param[i] < Fnm):
            temp.append('+')
        else:
            temp.append('-')
    
    # rss = calc_RSS(sigma_kv, n, len(teta))
    # rssh = calc_RSSH(cx, y, teta) 
    # F_zn_hip = calc_zn_hip(rss, rssh)

    df = pd.DataFrame()
    df['j'] = np.arange(len(teta))
    df['Верхнее значение'] = D_verh
    df['Нижнее значение'] = D_nizh
    df['Тетта задаваемое'] = teta
    df['Оценка тетта'] = mnk_teta

    df1 = pd.DataFrame()
    df1['F'] = F_zn_param
    df1['Отверг'] = temp

    create_table(df1)


    print('teta oz = ', mnk_teta)
    print('F = ', F )
    print('Ft = ', Ft)
    plot.show()

__main__()