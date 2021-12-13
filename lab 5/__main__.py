import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import linalg, sqrt
import scipy.stats
import pandas as pd
from scipy.stats import t as Student
import csv
import math

Fs = Student.ppf(1 - 0.05/2, 96) 

# Регрессия на 9 факторах. Эффект мультиколлинеарности создают две тройки факторов. Разброса в масштабах факторов нет. 

def u(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 

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

def create_x(x1, x2, x3, x4, x5, x6, x7, x8, x9, n):
    cx = np.zeros((len(x1), n))
    cx[:, 0] = 1
    cx[:, 1] = x2
    cx[:, 2] = x3
    cx[:, 3] = x4
    cx[:, 4] = x5
    cx[:, 5] = x6
    cx[:, 6] = x7
    cx[:, 7] = x8
    cx[:, 8] = x9
    return cx

def calc_det_inf_matrix(X):
    inf_matrix = np.dot(np.transpose(X), X)
    return np.linalg.det(inf_matrix)

def calc_det_inf_trace_matrix(X):
    inf_matrix = np.dot(np.transpose(X), X)
    tr = np.trace(inf_matrix)
    return np.linalg.det(np.divide(inf_matrix, tr))

def min_lambda(X):
    inf_matrix = np.dot(np.transpose(X), X)
    w = np.linalg.eigvals(inf_matrix)
    return min(w), w

def min_lambda_trace(X):
    inf_matrix = np.dot(np.transpose(X), X)
    tr = np.trace(inf_matrix)
    wtr= np.linalg.eigvals(np.divide(inf_matrix, tr))
    return min(wtr), wtr

def max_min_otn(w):
    return max(w) / min(w)

# Максимальная сопряженность, максимальная парная сопряженность
def matR(X):
    n = len(X)
    m = len(X[0])
    R = np.eye(m)
    Xx = [0]*m
    for i in range(m):
        for j in range(n):
            Xx[i] += X[j][i]**2 

    for i in range(m-1):
        for j in range(i+1, m):
            for k in range(n):
                R[i,j] += X[k][i] * X[k][j] / sqrt(Xx[i]*Xx[j])
                R[j,i] = R[i,j]
    maxij = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (i == j): continue
            if (maxij < R[i,j]):
                maxij = R[i,j]
    print("Max(Rij) = ", round(maxij,5)) 

    Rstar = np.transpose(R)
    Rstarinv = np.linalg.inv(Rstar)
    Rt = []
    for i in range(R.shape[0]):
        Rt.append(sqrt(1 - 1/(Rstarinv[i,i])))
    
    print("Max |Ri| = ", round(max(Rt),5))

def findMinEigIndex(vec):
    minIndex = 0
    value = vec[0]
    for i in range (1,len(vec)):
        if (value > vec[i]):
            minIndex = i
            value = vec[i]
    return minIndex

def MGK(X, Y, numberOfEigsToDelete):
    #переход к центрированным оценкам
    yavg = sum(Y)/len(Y)
    Ynew = Y
    for i in range(len(Y)):
        Ynew[i] = Ynew[i] - yavg
        
        
    Xnew = X
    for i in range(X.shape[1]):
        xavg = sum(X[:,i])/X.shape[0]
        for k in range(X.shape[0]):
            Xnew[k,i] = Xnew[k,i] - xavg
            
            
    XtXzvezd = np.dot(np.transpose(Xnew), Xnew)
    eigVals, V = np.linalg.eig(XtXzvezd)
    Z = np.dot(Xnew,V)
    #eigVals = np.linalg.eig(XtXzvezd)[0]
    Zr = Z
    Vr = V
    print("С.З - ", eigVals)
    for i in range(numberOfEigsToDelete):
        indexToDelete = findMinEigIndex(eigVals)
        Zr = np.delete(Zr, indexToDelete,1)
        Vr = np.delete(Vr, indexToDelete,1)
        eigVals = np.delete(eigVals, indexToDelete, 0)
        
    b = np.dot(np.transpose(Zr),Zr)
    b = np.linalg.inv(b)    
    b = np.dot(b, np.transpose(Zr))
    b = np.dot(b, Ynew)
    tetha = np.dot(Vr, b)
    
    RSS = Y - np.dot(X,tetha)
    RSS = np.dot(np.transpose(RSS), RSS)   
    return tetha,RSS

def MNK(X,lam,y):
    Xt = np.transpose(X)
    XtX = np.dot(Xt,X)
    tetha = np.linalg.inv(np.add(XtX, lam))
    tetha = np.dot(tetha, Xt)
    tetha = np.dot(tetha, y)
    return tetha

def calc_Ridge(X,Y, eigvals, lam):
    l = eigvals
    XtX = np.dot(np.transpose(X),X)
    LAMBDA = np.zeros((len(l),len(l)))
    for q in range(len(l)):
       LAMBDA[q,q] = lam * XtX[q,q]
    tetha = MNK(X, LAMBDA, Y)
    RSS = Y - np.dot(X,tetha)
    RSS = np.dot(np.transpose(RSS), RSS)
    return tetha,RSS

def create_plot_Ridge(X,Y, eigvals):
    fig, sbplt = plt.subplots(2, figsize =(5,5))
    x = np.arange(0, 0.03, 0.0001)
    y = []
    yy = []
    for i in x:
        #Построим матрицу LAMBDA
        tetha,RSS = calc_Ridge(X,Y, eigvals, i)
        y.append(RSS)
        yy.append(np.linalg.norm(tetha)**2)
    sbplt[0].plot(x,y)
    sbplt[1].plot(x,yy)    
    
def create_table(df):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')   
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()

def main():
    teta = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    n = 300
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)
    x4 = np.zeros(n)
    x5 = np.zeros(n)
    x6 = np.zeros(n)
    x7 = np.zeros(n)
    x8 = np.zeros(n)
    x9 = np.zeros(n)
    for i in range(n):
        x1[i] = random.uniform(-1, 1)
        x2[i] = random.uniform(-1, 1)
        x3[i] = x1[i] + x2[i] + np.random.normal(0,0.0001)
        x4[i] = random.uniform(-1, 1)
        x5[i] = random.uniform(-1, 1)
        x6[i] = random.uniform(-1, 1)
        x7[i] = x5[i] + x6[i] + np.random.normal(0,0.0001)
        x8[i] = random.uniform(-1, 1)
        x9[i] = random.uniform(-1, 1) 

    U = np.zeros(n)
    U = u(x1, x2, x3, x4, x5, x6, x7, x8, x9)
    X = create_x(x1, x2, x3, x4, x5, x6, x7, x8, x9, 9)
    y, e = calc_y(U)
    print('det=', calc_det_inf_matrix(X))
    print('detr=', calc_det_inf_trace_matrix(X))
    l, w = min_lambda(X)
    print('l=', l)
    lr, wr = min_lambda_trace(X)
    print('lr=', lr)
    r = max_min_otn(w)
    print('r=', r)
    matR(X)


    create_plot_Ridge(X, y, w)
    lam = 0.05 
    tetha,RSS = calc_Ridge(X, y, w, lam)
    print("Вектор tetha от ридж-оценок с lambda =", lam," -", tetha)
    print("RSS от ридж-оценок -", round(RSS,3))
    print("Норма вектора tetha =", round(np.linalg.norm(teta - tetha),3))

    #Метод главных компонент

    tetha,RSS = MGK(X[:, 1:], y , 2)
    print("Вектор tetha от главных компонент - ",tetha) 
    print("RSS от главных компонент =", round(RSS,3))
    print("Норма вектора tetha =", round(np.linalg.norm([1, 1, 1, 1, 1, 1, 1, 1] - tetha),3))
    # plt.show()

main()