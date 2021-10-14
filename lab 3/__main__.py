import numpy as np
import random
import matplotlib.pyplot as plot
from numpy import linalg as LA, sqrt
import scipy.stats
import pandas as pd
from scipy.stats import t as Student
import csv
from scipy.stats import f as Fisher

Fs = Student.ppf(1 - 0.05/2, 35) 

def u(x1, x2, teta):
    x1 = np.array(x1)
    x2 = np.array(x2)
    f = [1, 1 / x1, x2**2, x1*x2, x2 / x1]
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
        u_sr += (res - m)**2
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
    cx[:, 2] = x2**2
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
        F.append((teta[i])**2 / (sigma * djj[i]))
    return F

def calc_RSS(x, y, teta):
    rss = np.dot(np.transpose(y - np.dot(x, teta)), y - np.dot(x, teta))
    return rss

def calc_RSSH(x, y, teta):
    rssh = 0
    for i in range(len(y)): 
        rssh += (y[i] - np.average(y))**2
    return rssh

def calc_zn_hip(rss, rssh, n, m):
    F = ((rssh - rss) / (m-1)) / (rss / (n-m))
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

def read_value():
    with open('results.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        dataForAnalys = np.array(list(csv_reader))
        np.delete(dataForAnalys, (0), axis=0)
        x1 = dataForAnalys[:, 0]
        x1 = np.array([float(x) for x in x1])
        x2 = dataForAnalys[:, 1]
        x2 = np.array([float(x) for x in x2])
        y = dataForAnalys[:, 3]
        y = np.array([float(y) for y in y])
        e = dataForAnalys[:, 4]
        e = np.array([float(e) for e in e])
        return x1, x2, e, y

f = lambda x1, x2: [1, 1/x1, x2**2, x1*x2, x1 **2]

def calc_M_dov(u_oz, x, y, x1, x2, sigma, sigma_kv, teta):
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    M_verh = []
    M_nizh = []
    otkl_nizh = []
    otkl_verh = []
    u_dov = u(x1, x2, teta)
    y, e = calc_y(u_dov)
    for i in range(len(u_oz)): 
        X = LA.inv(np.dot(x.T, x))
        # fx = np.array(f(x1[i], 0))
        fx = np.array(f(1, x2[i]))
        vkl1 = np.matmul(fx.transpose(), X)
        vkl2 = vkl1@fx.T
        vkl = sigma * np.sqrt(vkl2)
        M_nizh.append(u_dov[i] - (Fs * vkl))
        M_verh.append(u_dov[i] + (Fs * vkl))

        vkl_otkl = sigma_kv * (1 + vkl2)
        otkl_nizh.append(u_dov[i] - (Fs * vkl_otkl))
        otkl_verh.append(u_dov[i] + (Fs * vkl_otkl))

    return M_verh, M_nizh, otkl_verh, otkl_nizh, u_dov, y

def create_plot(data1, data2, data3, x):
    fig, ax = plot.subplots()
    ax.set_title("Доверительный интервал для математичсекого ожидания.")
    # ax.set_title("Доверительный интервал для отклика.")
    ax.plot(x, data1, label='Левая граница')
    ax.plot(x, data2, label='Теоретическое значение')
    ax.plot(x, data3, label='Правая граница')
    ax.set_ylabel("у")
    ax.set_xlabel("х")
    ax.legend()
    ax.grid()

    fig.set_figheight(5)
    fig.set_figwidth(16)
    plot.show()
    plot.show()

def main():
    teta = [0.3, 0.14, 0.432, 3, 0.00001]
    n = 40
    x1, x2, e, y = read_value()
    U = np.zeros(40)
    U = u(x1, x2, teta)

    mnk_teta, cx = mnk(x1, x2, y, 5)
    e_oz = y - np.dot(cx, mnk_teta)
    y_oz = np.dot(cx, mnk_teta)

    sigma_kv = np.dot(e_oz.transpose(), e_oz)/ (n - len(teta))
    sigma = np.dot(e.transpose(),e) / (n - len(teta))
    print ('sigma kv=', sigma_kv)
    print ('sigma=', sigma)
    F = sigma_kv/sigma

    # Доверительный интервал
    Ft = scipy.stats.t.ppf(0.95, 36)
    djj = calc_djj(cx)
    D_verh, D_nizh = create_dov_int(cx, Ft, mnk_teta, sigma_kv, djj)
    df = pd.DataFrame()
    df['Верхнее значение'] = D_verh
    df['Нижнее значение'] = D_nizh
    df['Тетта задаваемое'] = teta
    df['Оценка тетта'] = mnk_teta

# Вычисление гипотезы о незначимости параметров
    F_zn_param = calc_neznach_hipot_param(mnk_teta, djj, sigma_kv)
    temp = []
    F_fish = scipy.stats.f.ppf(1-0.05/2, 1, 35)
    for i in range(len(mnk_teta)):
        if(F_zn_param[i] < F_fish):
            temp.append('Не отвергается')
        else: 
            temp.append('Отвергается')
    df1 = pd.DataFrame()
    df1['F'] = F_zn_param
    df1['Оценка параметра'] = mnk_teta
    df1['Отверг'] = temp
    print(F_fish)

    # Вычисление гипотезы о незначимости гипотезы 
    temp1 = []
    rss = calc_RSS(cx, y, teta)
    rssh = calc_RSSH(cx, y, teta)
    print('rss=', rss) 
    print('rssh=', rssh) 
    F_zn_hip = calc_zn_hip(rss, rssh, n, len(teta))
    print('F_zn_hip=', F_zn_hip)
    F_fish = Fisher.ppf(1-0.05/2, 1, 35)
    print('F_fish=', F_fish)
    if F_zn_hip < F_fish: 
        temp1.append('Не отвергается')
    else: 
        temp1.append('Отвергается')
    df2 = pd.DataFrame()
    df2['F'] = pd.Series(F_zn_hip)
    df2['Квантиль F распределения'] = F_fish
    df2['Гипотеза'] = temp1

    # Вычисление дов интервала мат ожиадния
    u_M = u(x1, x2, mnk_teta)
    M_teir = u(x1, x2, teta)
    M_verh, M_nizh, otkl_ver, otkl_nizh, u_dov, y = calc_M_dov(u_M, cx, y, x1, x2, sigma, sigma_kv, teta)

    # create_table(df)
    # create_table(df1)
    # create_table(df2)
    # create_plot(M_nizh, u_dov, M_verh, np.sort(x1))
    # create_plot(otkl_nizh, u_dov, otkl_ver, np.sort(x1))

    create_plot(M_nizh, u_dov, M_verh, np.sort(x2))
    # create_plot(otkl_nizh, u_dov, otkl_ver, np.sort(x2))

    print('teta oz = ', np.round(mnk_teta, 5))
    print('F = ', F )
    print('Ft = ', Ft)
    plot.show()

main()