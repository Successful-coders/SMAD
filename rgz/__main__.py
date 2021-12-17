import csv
import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt

def read_value():
    with open('input_rgz.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        dataForAnalys = np.array(list(csv_reader))
        np.delete(dataForAnalys, (0), axis=0)
        x1 = dataForAnalys[:, 0]
        x1 = np.array([float(x) for x in x1])
        x2 = dataForAnalys[:, 1]
        x2 = np.array([float(x) for x in x2])
        x3 = dataForAnalys[:, 3]
        x3 = np.array([float(x) for x in x3])
        x4 = dataForAnalys[:, 4]
        x4 = np.array([float(x) for x in x4])
        y = dataForAnalys[:, 4]
        y = np.array([float(y) for y in y])
        return x1, x2, x3, x4, y

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

def create_x(x1, x2, x3, x4, n):
    cx = np.zeros((len(x1), n))
    cx[:, 0] = x1
    cx[:, 1] = x2
    cx[:, 2] = x3
    cx[:, 3] = x4
    return cx

def MNK(cx, cy):
    teta = (np.dot(cx.transpose(), cx))
    teta = LA.inv(teta)
    teta = np.dot(teta, cx.transpose())
    teta = np.dot(teta, cy)
    return teta

def calc_d_vozm(x1, x2, x3, x4):
    d = []
    for i in range (len(x1)):
        d.append(math.sqrt((x1[i]-1) ** 2 + x2[i] ** 2))
    return d

def calc_V(disp, n):
    V = np.zeros((n, n))
    for i in range(n):
        V[i][i] = disp[i]    
    return V

def multikoliniar_test(X):
    # print('det=', calc_det_inf_matrix(X))
    print('detr=', calc_det_inf_trace_matrix(X))
    l, w = min_lambda(X)
    # print('l=', l)
    lr, wr = min_lambda_trace(X)
    print('lr=', lr)
    r = max_min_otn(w)
    print('r=', r)

def calc_m(u):
    sum = np.sum(u)
    return sum/len(u)

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

def u_ed(x1, x2, x3, x4, teta):
    return teta[0]*x3 + teta[1]* (x1) + teta[2]*x2**2 + teta[3]* x1* x2 + x4


def test_goldfelda(X, d, N, u, teta):
    #    Тест №2 - Голдфельда-Квандтона
    #    Упорядочим матрицу наблюдений по величине модуля суммы двух факторов

    XwDisp = np.array(X)
    d = np.reshape(d,(N,1))
    XwDisp = np.append(XwDisp, d, axis=1)
    u = np.reshape(u,(N,1))
    XwDisp = np.append(XwDisp, u, axis=1) 

    Xsorted = np.array(sorted(XwDisp, key=lambda a: abs(u_ed(a[1], a[2], a[3], a[4], teta))))
    nc = int(N/3)
    X1 = Xsorted[:nc]
    X2 = Xsorted[2*nc:]

    tetha = MNK(X1[:,:4], X1[:,5])
    e1 = X1[:,4] - np.dot(X1[:,:4],tetha)

    tetha = MNK(X2[:,:4], X2[:,5])
    e2 = X2[:,4] - np.dot(X2[:,:4],tetha)

    print("RSS2/RSS1 = ",(np.dot(np.transpose(e2), e2))/(np.dot(np.transpose(e1), e1)))

def autokor(X, y, pk, n):
    #проверка на автокорреляцию
    dl = 1.55
    du = 1.75
    print('\nПроверка данных на автокорреляцию')
    et = y - (pk.T@X.T).T
    dw = np.array([(et[t] - et[t-1])**2 for t in range(n)]).sum() / (et**2).sum()
    if dw < dl:
        print('DW =', dw, ' - положительная автокорреляция')
    else:
        if dw > du and dw < 4 - dl:
            print('DW =', dw, ' - отсутствие автокорреляции')
        else:
            if dw > 4 - dl:
                print('DW =', dw, ' - отрицательная автокорреляция')
    p = dw


def mnk_z(Z, c):
    # Оценивание параметров alpha через обыкновенный МНК (XtX)-1*Xt*Y
    Zt = np.transpose(Z)
    skobka = np.dot(Zt,Z)
    skobka = np.linalg.matrix_power(skobka, -1)
    skobka = np.dot(skobka,Zt)
    resZ = np.dot(skobka, c)
    return resZ

def create_z(d):
    Z = np.array([0,0])
    for i in range(len(d)):
        newrow = [1, d[i]]    
        Z = np.vstack([Z, newrow])
    Z = np.delete(Z, 0, axis = 0) 
    return Z

def u(x1, x2, x3, x4, teta):
    x1 = np.array(x1)
    x2 = np.array(x2)
    f = [x1, x2**2, x3, x4]
    nu = np.dot(teta, f)
    return nu
#ОМНК
def OMNK(X, p, y):
    n = len(X)
    V = np.zeros((n,n))
    i = 0
    for i in range(n-1):
        V[i][i+1] = -p
        V[i][i] = 1+p**2
        V[i+1][i] = -p
        i = i + 1
    V[0][0] = 1
    V[n-1][n-1] = 1

    pk= X.T@V
    pk= np.linalg.inv((X.T@V)@X)
    pk= ((pk@X.T)@V)@y
    return pk

def kox(reg, n, y):
    model = np.ones((n))
    model = np.column_stack((model, reg[0]))
    model = np.column_stack((model, reg[1]))
    model = np.column_stack((model, reg[11]))
    model = np.column_stack((model, reg[12]))

    #оценка через процедуру Кохрейна-Оркатта
    print('\nОценка параметров лучшей модели процедурой Кохрейна-Оркатта:')
    p=0
    while 1:
        pk = OMNK(X=model, p=p, y)
        #pk = ((np.linalg.inv(model.T@model))@model.T)@(y)
        et = y - (pk.T@model.T).T
        p_prev = p
        p = ((np.linalg.inv(et[:n-1].T@et[:n-1]))@et[:n-1].T)@(et[1:n])
        print(p)
        if np.abs(p_prev - p) < 0.001:
            break
    pk= OMNK(X=model, p=p, y)
    print('p =', p)
    print('pk = ')
    print(pk.T)

def dep_X_Y(X, y):
    reg = {
        0: X[:, 0], #x1
        1: X[:, 1], #x2
        2: X[:, 2], #x3
        3: X[:, 3], #x4
        
        4: X[:, 0] * X[:, 0], #x1x1
        5: X[:, 0] * X[:, 1], #x1x2
        6: X[:, 0] * X[:, 2], #x1x3
        7: X[:, 0] * X[:, 3], #x1x4
        
        8:  X[:, 1] * X[:, 1], #x2x2
        9:  X[:, 1] * X[:, 2], #x2x3
        10: X[:, 1] * X[:, 3], #x2x4    
        
        11: X[:, 2] * X[:, 2], #x3x3
        12: X[:, 2] * X[:, 3], #x3x4

        13: X[:, 3] * X[:, 3], #x4x4
        14: X[:, 0] * X[:, 1]* X[:, 2], #x1x2x3
        15: X[:, 0] * X[:, 1]* X[:, 3], #x1x2x4
        16: X[:, 0] * X[:, 2]* X[:, 3], #x1x3x4
        17: X[:, 1] * X[:, 2]* X[:, 3], #x2x3x4
    }
    data = {
        0: 'X1',
        1: 'X2',
        2: 'X3',
        3: 'X4',
        
        4: 'X1X1',
        5: 'X1X2',
        6: 'X1X3',
        7: 'X1X4',
        
        8:  'X2X2',
        9:  'X2X3',
        10: 'X2X4',
        
        11: 'X3X3',
        12: 'X3X4',

        13: 'X4X4',
        
        14: 'X1X2X3',
        15: 'X1X2X4',
        16: 'X1X3X4',
        17: 'X2X3X4',
    }

    
    # for i in range(0, len(data)):
    #     p = data[i]
    #     x = reg[i]
    #     plt.scatter(reg[i], y)
    #     plt.title(f'Зависимость Y от {p}')
    #     plt.xlabel(f'{p}')
    #     plt.ylabel('y')
    #     plt.show()
    return reg

def main():
    x1, x2, x3, x4, y = read_value()
    X = create_x(x1, x2, x3, x4, 4)
    multikoliniar_test(X)
    teta = MNK(X, y)
    print(teta)
    d = calc_d_vozm(x1, x2, x3, x4)
    e_oz = y - np.dot(X, teta)
    sigmaTheoretical = (np.dot(np.transpose(e_oz), e_oz)) / len(X)
    sigma = (e_oz * e_oz)/sigmaTheoretical
    Z = create_z(d)
    U = u(x1, x2, x3, x4, teta)
    # обычный мнк c Z
    resZ = mnk_z(Z, sigma)
    
    # test 1
    sigma_oz = test_breusha_pagana(sigma, resZ, Z)

    # test 2
    res = test_goldfelda(X, d, len(X), U, teta)
    # автокореляция
    pk = OMNK(X, 0, y)
    autokor(X, y, pk, len(X))
    reg = dep_X_Y(X, y)
    kox(reg, len(X), y)

main()