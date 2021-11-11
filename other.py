import random
import numpy  
from math import *

N = 300

trueTetha = [3, 2, 0.001, 4, 0.0015, 3, 0.002, 4, 2]
def u(x1,x2,x3):
    return trueTetha[0]*1 + trueTetha[1]*x1 + trueTetha[2]*x1**2 + trueTetha[3]*x2 + trueTetha[4]*x2**2 + trueTetha[5]*x3 + trueTetha[6]*x3**2 + trueTetha[7]*x1*x2 + trueTetha[8]*x1*x3 

def dispers(x1,x2,x3):
    return 0.05+0.01*abs(u(x1,x2,x3))

def findMx(reses):
    Mx = 0
    for i in reses:
        Mx += i
    Mx /= len(reses)
    return Mx

def makeOmega(X, func, alpha = 1):
#     Функция создания матрицы Омега большое, Х - матрица наблюдений (1 - первый фактор, 3 - второй, 5 - третий)
    Omega = numpy.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        Omega[i,i] = 1/(func(X[i,1],X[i,3],X[i,5])*alpha)
    return Omega

def MNK(X, Y, func = lambda x1,x2,x3 : 1, alpha = 1):
    
    #    Обобщенный МНК
    #    Х - матрица наблюдений
    #    Y - матрица откликов, соответствующая Х
    #    func - функция h (дефолт - 1, дисперсия - константа)
    
    Omega = makeOmega(X, func, alpha)
    Xt = numpy.transpose(X)
    skobka= numpy.dot(Xt,Omega)
    skobka = numpy.dot(skobka,X)
    skobka = numpy.linalg.matrix_power(skobka, -1)
    skobka = numpy.dot(skobka,Xt)
    skobka = numpy.dot(skobka,Omega)
    tetha = numpy.dot(skobka, Y)
    return tetha

#     Генерация точек 

random.seed(1)
results = []
for i in range (N):
    x1 = round(random.uniform(-1,1),3)
    x2 = round(random.uniform(-1,1),3)
    x3 = round(random.uniform(-1,1),3)
    res = round(u(x1,x2,x3),3)
    results.append([x1,x2,x3,res])
 
#    Генерация шума для каждого значения 
for i in range(len(results)):
    shum = numpy.random.normal(0,dispers(results[i][0], results[i][1], results[i][2])**(1/2))
    results[i].append(round(shum,3))
    results[i].append(round(results[i][3] + results[i][4],3))
    

 
#    Тест №1 - Бреуша-Пагана 
# Создаем матрицы регрессоров, а также вектор откликов
X = numpy.array([0,0,0,0,0,0,0,0,0])
Y = []
for i in range(len(results)):
    x1 = results[i][0]
    x2 = results[i][1]
    x3 = results[i][2]
    newrow = [1,x1,x1**2, x2, x2**2, x3, x3**2, x1*x2, x1*x3]    
    X = numpy.vstack([X, newrow])
    Y.append(results[i][5])
X = numpy.delete(X, 0, axis = 0)

#    Оценивание параметров tetha через обыкновенный МНК
tetha = MNK(X,Y)
print('Вектор параметров tetha для всей выборки - ', tetha)
d2 = 0
for i in range(len(tetha)):
    d2 += (tetha[i] - trueTetha[i])**2
print("Расстояние tetha  - ", d2)
e = numpy.subtract(Y,numpy.dot(X,tetha))
sigmaTheoretical = (numpy.dot(numpy.transpose(e), e)) / (N)
print("Сигма оценочная - ",sigmaTheoretical)
c = (e*e)/sigmaTheoretical

Z = numpy.array([0,0])
for i in range(len(results)):
    x1 = results[i][0]
    x2 = results[i][1]
    x3 = results[i][2]
    newrow = [1, dispers(x1,x2,x3)]    
    Z = numpy.vstack([Z, newrow])
Z = numpy.delete(Z, 0, axis = 0) 


#    Оценивание параметров alpha через обыкновенный МНК (XtX)-1*Xt*Y
Zt = numpy.transpose(Z)
skobka = numpy.dot(Zt,Z)
skobka = numpy.linalg.matrix_power(skobka, -1)
skobka = numpy.dot(skobka,Zt)
resZ = numpy.dot(skobka, c)
print('Вектор параметров alpha - ',resZ) 

 
#    Подсчет ESS, сравнение с критической статистикой
resZ = numpy.delete(resZ,0,0) # удаляем свободный член
Z = numpy.delete(Z,0,1) # здесь также удаляем свободный член
cc = numpy.dot(resZ, numpy.transpose(Z))
ESS = 0
avg = findMx(c)
for i in range(len(c)):
    ESS += (cc[i] - avg)**2
print(ESS/2," ? ",  3.84)

#    Тест №2 - Голдфельда-Квандтона
#    Упорядочим матрицу наблюдений по величине модуля суммы двух факторов

XwDisp = numpy.array(X)
results = numpy.array(results)
tmp = results[:,3]
tmp = numpy.reshape(tmp,(N,1))
XwDisp = numpy.append(XwDisp, tmp, axis=1)

tmp = results[:,5]
tmp = numpy.reshape(tmp,(N,1))
XwDisp = numpy.append(XwDisp, tmp, axis=1) 

Xsorted = numpy.array(sorted(XwDisp, key=lambda a: abs(u(a[1],a[3],a[5]))))
nc = int(N/3)
X1 = Xsorted[:nc]
X2 = Xsorted[2*nc:]
tetha = MNK(X1[:,:9], X1[:,10])
print('Вектор параметров tetha X1 - ',tetha)
e1 = X1[:,9] - numpy.dot(X1[:,:9],tetha)
d2 = 0
for i in range(len(tetha)):
    d2 += (tetha[i] - trueTetha[i])**2
print("Расстояние X1 - ", d2)

tetha = MNK(X2[:,:9], X2[:,10])
print('Вектор параметров tetha X2 - ',tetha)
e2 = X2[:,9] - numpy.dot(X2[:,:9],tetha)
d2 = 0
for i in range(len(tetha)):
    d2 += (tetha[i] - trueTetha[i])**2
print("Расстояние X2 - ", d2)

print("RSS2/RSS1 = ",(numpy.dot(numpy.transpose(e2), e2))/(numpy.dot(numpy.transpose(e1), e1)))


tetha = MNK(X1[:,:9],X1[:,10],dispers, resZ)
print('Вектор параметров tetha X1 (корр.) - ',tetha)
d2 = 0
for i in range(len(tetha)):
    d2 += (tetha[i] - trueTetha[i])**2
print("Расстояние X1 (корр. - ", d2)

tetha = MNK(X2[:,:9], X2[:,10],dispers, resZ)
print('Вектор параметров tetha X2 (корр.) - ',tetha)
d2 = 0
for i in range(len(tetha)):
    d2 += (tetha[i] - trueTetha[i])**2
print("Расстояние X2 (корр.) - ", d2)

#   Обобщенный МНК

tetha = MNK(X, Y,dispers, resZ)
print('Вектор параметров tetha (корр.) - ',tetha)
d2 = 0
for i in range(len(tetha)):
    d2 += (tetha[i] - trueTetha[i])**2
print("Расстояние общей выборки - ", d2)
