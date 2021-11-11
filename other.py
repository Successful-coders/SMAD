import random
import numpy  
from math import *
N = 300
import matplotlib.pyplot as plt

def u(x1,x2,x3,x4,x5,x6,x7,x8, x9):
    return 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9

def fMx(res):
    Mx=0
    for i in res:
        Mx += i
    Mx /= len(res)
    return Mx
 
    
def fDx(res):
     Dx = 0
     Mx = fMx(res)
     for i in res:
         Dx += (i - Mx)**2
     Dx = Dx / (len(res) - 1)
     return Dx


# Максимальная сопряженность, максимальная парная сопряженность
def matR(X):
    n = len(X)
    m = len(X[0])
    R = numpy.eye(m)
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
    Rstar = numpy.transpose(R)
    Rstarinv = numpy.linalg.inv(Rstar)
    Rt = []
    for i in range(R.shape[0]):
        Rt.append(sqrt(1 - 1/(Rstarinv[i,i])))
    
    print("Max |Ri| = ", round(max(Rt),5))
 
def MNK(X,lam,y):
    Xt = numpy.transpose(X)
    XtX = numpy.dot(Xt,X)
    tetha = numpy.linalg.inv(numpy.add(XtX, lam))
    tetha = numpy.dot(tetha, Xt)
    tetha = numpy.dot(tetha, y)
    return tetha

def makeRidge(X,Y, eigvals, lam):
    l = eigvals
    XtX = numpy.dot(numpy.transpose(X),X)
    LAMBDA = numpy.zeros((len(l),len(l)))
    for q in range(len(l)):
       LAMBDA[q,q] = lam * XtX[q,q]
    tetha = MNK(X, LAMBDA, Y)
    RSS = Y - numpy.dot(X,tetha)
    RSS = numpy.dot(numpy.transpose(RSS), RSS)
    return tetha,RSS
        
def plotRidge(X,Y, eigvals):
    fig, sbplt = plt.subplots(2, figsize =(5,5))
    x = numpy.arange(0, 0.03, 0.0001)
    y = []
    yy = []
    for i in x:
        #Построим матрицу LAMBDA
        tetha,RSS = makeRidge(X,Y, eigvals, i)
        y.append(RSS)
        yy.append(numpy.linalg.norm(tetha)**2)
    sbplt[0].plot(x,y)
    sbplt[1].plot(x,yy)    
    
    
def findMinEigIndex(vec):
    minIndex = 0
    value = vec[0]
    for i in range (1,len(vec)):
        if (value > vec[i]):
            minIndex = i
            value = vec[i]
    return minIndex

def MGK(X,Y, numberOfEigsToDelete):
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
            
            
    XtXzvezd = numpy.dot(numpy.transpose(Xnew), Xnew)
    eigVals, V = numpy.linalg.eig(XtXzvezd)
    Z = numpy.dot(Xnew,V)
    #eigVals = numpy.linalg.eig(XtXzvezd)[0]
    Zr = Z
    Vr = V
    print("С.З - ", eigVals)
    for i in range(numberOfEigsToDelete):
        indexToDelete = findMinEigIndex(eigVals)
        Zr = numpy.delete(Zr, indexToDelete,1)
        Vr = numpy.delete(Vr, indexToDelete,1)
        eigVals = numpy.delete(eigVals, indexToDelete, 0)
        
    b = numpy.dot(numpy.transpose(Zr),Zr)
    b = numpy.linalg.inv(b)    
    b = numpy.dot(b, numpy.transpose(Zr))
    b = numpy.dot(b, Ynew)
    tetha = numpy.dot(Vr, b)
    
    RSS = Y - numpy.dot(X,tetha)
    RSS = numpy.dot(numpy.transpose(RSS), RSS)   
    return tetha,RSS

    
#     Генерация точек 

random.seed(1)
U = []
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
X6 = []
X7 = []
X8 = []
X9 = []
for i in range (N):
    x1 = round(random.uniform(-1,1),3)
    x2 = round(random.uniform(-1,1),3)
    x3 = round(random.uniform(-1,1),3)
    x4 = round(random.uniform(-1,1),3)
    x5 = round(random.uniform(-1,1),3)
    x6 = round(random.uniform(-1,1),3)
    x7 = round(random.uniform(-1,1),3)
    x8 = round(random.uniform(-1,1),3)
    x9 = round(2*x1*x2 *x5 + x6 *10*x3 *x4 + numpy.random.normal(0,0.1),3)
    res = round(u(x1,x2,x3,x4,x5,x6,x7,x8, x9),3)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)
    X4.append(x4)
    X5.append(x5)
    X6.append(x6)
    X7.append(x7)
    X8.append(x8)
    X9.append(x9)
    U.append(res)
    
    
 
#Получение значений шума е и зашумленного значения функции у
y=[] 
e=[]   
w2 = fDx(U)     
sigma = sqrt(0.05*(w2))
for i in range(len(U)):
    e.append(round(numpy.random.normal(0, sigma),5))
    y.append(round((U[i] + e[i]),5))
    
    #запись в файл результатов генерирования
f = open("results.txt", 'w')
for i in range(len(y)):
    res = ''
    res += '('+ str(X1[i]) + ', ' + str(X2[i]) + ', ' + str(X3[i]) +', ' + str(X4[i]) + ', ' + str(X5[i]) + ', ' + str(X6[i]) + ', ' + str(X7[i]) + ', ' + str(X8[i]) +'); '
    res += str(U[i]) + '; ' + str(e[i]) + '; ' + str(y[i]) 
    res += '\n'
    f.write(res)
f.close()

#Матрицы регрессоров и  вектор откликов
X = numpy.array([0,0,0,0,0,0,0,0,0,0])
Y = []
for i in range(len(X1)):
    x1 = X1[i]
    x2 = X2[i]
    x3 = X3[i]
    x4 = X4[i]
    x5 = X5[i]
    x6 = X6[i]
    x7 = X7[i]
    x8 = X8[i]
    x9 = X9[i]
    newrow = [1,x1,x2, x3, x4, x5, x6, x7, x8, x9]    
    X = numpy.vstack([X, newrow])
    Y.append(y[i])
    
    
X = numpy.delete(X, 0, axis = 0)

# Определитель матрицы ХтХ

Xt = numpy.transpose(X)
XtX = numpy.dot(Xt,X)
tr = numpy.trace(XtX)
    
    
print("Определитель матрицы ХтХ ",numpy.linalg.det(XtX))
print("Определитель матрицы ХтХ/trХтХ ",numpy.linalg.det(numpy.divide(XtX, tr)) )    

# Минимальное собственное число матрицы ХтХ

w = numpy.linalg.eigvals(XtX)
print("Минимальное собственное значение матрицы ХтХ ",min(w))
wtr = numpy.linalg.eigvals(numpy.divide(XtX, tr))
print("Минимальное собственное значение матрицы ХтХ/trХтХ ",min(wtr))

# Отношение max/min

print("max/min матрицы ХтХ", max(w)/min(w))


# Cопряженные и попарно сопряженные

matR(X)

# Ридж-оценки

plotRidge(X, y, w)
lam = .003  
tetha,RSS = makeRidge(X, y, w, lam)
print("Вектор tetha от ридж-оценок с lambda =", lam," -", tetha)
print("RSS от ридж-оценок -", round(RSS,3))
print("Норма вектора tetha =", round(numpy.linalg.norm([1,1,1,1,1,1,1,1,1,1] - tetha),3))

#Метод главных компонент

tetha,RSS = MGK(X[:, 1:], y , 2)
print("Вектор tetha от главных компонент - ",tetha) 
print("RSS от главных компонент =", round(RSS,3))
print("Норма вектора tetha =", round(numpy.linalg.norm([1,1,1,1,1,1,1,1,1] - tetha),3))
