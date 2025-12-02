import time
import numpy as np
import scipy.special
from itertools import combinations
from datetime import timedelta
from decimal import *
import xml.etree.ElementTree as ET
import sympy
import math
from functools import reduce
import mpmath
import requests
#import gmpy2
from numba import njit, prange

start_time = time.monotonic()

def binomial_mpmath(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Aprovechar la simetría
    numerator = mpmath.fac(n)
    denominator = mpmath.fac(k) * mpmath.fac(n - k)
    return numerator / denominator

def binomial(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Aprovechar la simetría
    numerator = reduce(lambda x, y: x * y, range(n, n - k, -1), 1)
    denominator = reduce(lambda x, y: x * y, range(1, k + 1), 1)
    return numerator // denominator

def result(lista, konbodanak):
    resultaue=["1","X","2"]
    plenue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    plen=["00","01","02","0M","10","11","12","1M","20","21","22","2M","M0","M1","M2","MM"]
    f= open("results/resul.txt", "w")
    for i in range(len(lista)):
        #print(lista[i][0])
        #print(resultaue[konbodanak[lista[i][1]][0]],resultaue[konbodanak[lista[i][1]][1]],resultaue[konbodanak[lista[i][1]][2]],resultaue[konbodanak[lista[i][1]][3]],resultaue[konbodanak[lista[i][1]][4]],resultaue[konbodanak[lista[i][1]][5]],resultaue[konbodanak[lista[i][1]][6]],resultaue[konbodanak[lista[i][1]][7]],resultaue[konbodanak[lista[i][1]][8]],resultaue[konbodanak[lista[i][1]][9]],resultaue[konbodanak[lista[i][1]][10]],resultaue[konbodanak[lista[i][1]][11]],resultaue[konbodanak[lista[i][1]][12]],resultaue[konbodanak[lista[i][1]][13]])
        #print(plenue[lista[i][2]])
        f.write(resultaue[konbodanak[lista[i][1]][0]]+resultaue[konbodanak[lista[i][1]][1]]+resultaue[konbodanak[lista[i][1]][2]]+resultaue[konbodanak[lista[i][1]][3]]+resultaue[konbodanak[lista[i][1]][4]]+resultaue[konbodanak[lista[i][1]][5]]+resultaue[konbodanak[lista[i][1]][6]]+resultaue[konbodanak[lista[i][1]][7]]+resultaue[konbodanak[lista[i][1]][8]]+resultaue[konbodanak[lista[i][1]][9]]+resultaue[konbodanak[lista[i][1]][10]]+resultaue[konbodanak[lista[i][1]][11]]+resultaue[konbodanak[lista[i][1]][12]]+resultaue[konbodanak[lista[i][1]][13]] + plen[lista[i][2]] + "\n")
    f.close()

def portz(lista):
    total=0.0
    for i in range(len(lista)):
            total+=1/lista[i]
    for i in range(len(lista)):
        lista[i]=1/(total*lista[i])


@njit
def calcular_producto(danakb, indices):
    prod = 1.0
    for i in indices:
        prod *= danakb[i]
    return prod

@njit(parallel=True)
def calcular_benetakue(danakb, konbin, sobrakx):
    benetakue = np.zeros(len(danakb))
    for i in prange(len(danakb)):
        for z in prange(len(konbin)):
            prod = calcular_producto(danakb[i], konbin[z])
            benetakue[i] += prod * (1 - danakb[i][sobrakx[z]])
    return benetakue     

@njit(parallel=True)
def calcular_benetakue2(danakb, konbin, sobrakx):
    benetakue = np.zeros(len(danakb))
    for i in prange(len(danakb)):
        for z in prange(len(konbin)):
            prod = calcular_producto(danakb[i], konbin[z])
            for s in sobrakx[z]:
                prod *= (1 - danakb[i][s])
            benetakue[i]+= prod
    return benetakue

bene=[]
lae=[]

with open("quiniela/estimacion.txt", mode="r", encoding="utf-8-sig") as datos:
    valores=[]
    for linea in datos:
        linea=linea.replace(",",".")
        valores.append([float(x)/100 for x in linea.split()])
for i in range(0,14):
    lae.append([valores[i][0],valores[i][1],valores[i][2]])
for i in range(14,15):
    lae.append([valores[i][0]*valores[i+1][0],valores[i][0]*valores[i+1][1],valores[i][0]*valores[i+1][2],valores[i][0]*valores[i+1][3],valores[i][1]*valores[i+1][0],valores[i][1]*valores[i+1][1],valores[i][1]*valores[i+1][2],valores[i][1]*valores[i+1][3],valores[i][2]*valores[i+1][0],valores[i][2]*valores[i+1][1],valores[i][2]*valores[i+1][2],valores[i][2]*valores[i+1][3],valores[i][3]*valores[i+1][0],valores[i][3]*valores[i+1][1],valores[i][3]*valores[i+1][2],valores[i][3]*valores[i+1][3]])

botie = 0
estimazixue = 851064
temporadie = 2026
jornadie = 25
plenuehasi= 8
plenuebukau= 9
datos=[34,23,26,34,17,10,13,13,17,10,11,12,13,7.5,8,8]
url = "https://api.eduardolosilla.es/servicios/v1/probabilidad_real?jornada="+str(jornadie)+"&temporada="+str(temporadie)
response = requests.get(url)
root = ET.fromstring(response.content)

 
#tree=ET.parse("bene.xml")
#root=tree.getroot()
for elem in root:
    for part in elem:
        if part.get("num")!="15":
            benetxiki=[float((part.get("porcDec_1")))/100]
            benetxiki.append(float((part.get("porcDec_X")))/100)
            benetxiki.append(float((part.get("porcDec_2")))/100)
            bene.append(benetxiki)
            benetxiki=[]
portz(datos)
bene.append(datos)

#76.527.504 konbinazixo
jendie= estimazixue/0.75
bukaera=[]

irabazixek10=[0] * 1000
irabazixek11=[0] * 1000
irabazixek12=[0] * 1000
irabazixek13=[0] * 1000
irabazixek14=[0] * 1000
irabazixek15=[0] * 1000
konbodanak=[]


sobrantie = botie-estimazixue*0.235 
if sobrantie < 0:
    sobrantie = 0

for yi in range(0,1000):
    estim=estimazixue*0.075
    if (estim+sobrantie)/(yi+1)>40000:
        irabazixek15[yi]=((estim+sobrantie)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek15[yi]=(estim+sobrantie)/(yi+1)
    if (estimazixue*0.16)/(yi+1)>40000:
        irabazixek14[yi]=((estimazixue*0.16)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek14[yi]=(estimazixue*0.16)/(yi+1)
    
    if (estim)/(yi+1)>40000:
        irabazixek13[yi]=((estim)/(yi+1)-40000)*0.8+40000
        irabazixek12[yi]=((estim)/(yi+1)-40000)*0.8+40000
        irabazixek11[yi]=((estim)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek13[yi]=(estim)/(yi+1)
        irabazixek12[yi]=(estim)/(yi+1)
        irabazixek11[yi]=(estim)/(yi+1)
    if (estimazixue*0.09)/(yi+1)>40000:
        irabazixek10[yi]=((estimazixue*0.09)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek10[yi]=(estimazixue*0.09)/(yi+1)
  
konbodanak=np.load("combinations/motza.npy")
print(len(konbodanak))     

cont=0
numeros=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
sobrakx=[13,12,11,10,9,8,7,6,5,4,3,2,1,0]
konbin = list(combinations(numeros, 13))
konbinazix = list(combinations(numeros, 12))
konbinazixuk = list(combinations(numeros, 11))
kbinazixuk = list(combinations(numeros, 10))
num = set(numeros)
sobrx=[]
sobx=[]
sox=[]
for elem in konbinazix:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobrx.append(faltantes)
for elem in konbinazixuk:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobx.append(faltantes)
for elem in kbinazixuk:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sox.append(faltantes) 

danakb = []
danakl = []
for xi in konbodanak:
    sub_b = []
    sub_l = []
    for i in range(0,14):
        sub_b.append(bene[i][xi[i]])
        sub_l.append(lae[i][xi[i]])
    danakb.append(sub_b)
    danakl.append(sub_l)

benetakue13 = calcular_benetakue(np.array(danakb), np.array(konbin), np.array(sobrakx))
benetakue12 = calcular_benetakue2(np.array(danakb), np.array(konbinazix), np.array(sobrx))
benetakue11 = calcular_benetakue2(np.array(danakb), np.array(konbinazixuk), np.array(sobx))
benetakue10 = calcular_benetakue2(np.array(danakb), np.array(kbinazixuk), np.array(sox))
laek13 = calcular_benetakue(np.array(danakl), np.array(konbin), np.array(sobrakx))
laek12 = calcular_benetakue2(np.array(danakl), np.array(konbinazix), np.array(sobrx))
laek11 = calcular_benetakue2(np.array(danakl), np.array(konbinazixuk), np.array(sobx))
laek10 = calcular_benetakue2(np.array(danakl), np.array(kbinazixuk), np.array(sox))

for i in range(len(danakb)):
    for g in range(plenuehasi,plenuebukau):
        benetakue14=(danakb[i][0])*(danakb[i][1])*(danakb[i][2])*(danakb[i][3])*(danakb[i][4])*(danakb[i][5])*(danakb[i][6])*(danakb[i][7])*(danakb[i][8])*(danakb[i][9])*(danakb[i][10])*(danakb[i][11])*(danakb[i][12])*(danakb[i][13])
        benetakue15= benetakue14 * (bene[14][g])
        laek14= (danakl[i][0])*(danakl[i][1])*(danakl[i][2])*(danakl[i][3])*(danakl[i][4])*(danakl[i][5])*(danakl[i][6])*(danakl[i][7])*(danakl[i][8])*(danakl[i][9])*(danakl[i][10])*(danakl[i][11])*(danakl[i][12])*(danakl[i][13])
        laek15 = laek14 * (lae[14][g])
        laekue15=0.0
        laekue14=0.0
        laekue13=0.0
        laekue12=0.0
        laekue11=0.0
        laekue10=0.0

        for y in range(0,50):
            #jendie+1 arte berez
            binomio=scipy.special.comb(jendie, y, exact=True)
            #binomio=gmpy2.comb(jendie,y)
            #binomio=sympy.binomial(jendie,y) 
            #binomio=binomial_mpmath(jendie,y)
            laekue10+=((1-laek10[i])**(jendie-y))*irabazixek10[y]*(laek10[i]**y)*binomio
            laekue11+=((1-laek11[i])**(jendie-y))*irabazixek11[y]*(laek11[i]**y)*binomio
            laekue12+=((1-laek12[i])**(jendie-y))*irabazixek12[y]*(laek12[i]**y)*binomio
            laekue13+=((1-laek13[i])**(jendie-y))*irabazixek13[y]*(laek13[i]**y)*binomio
            laekue14+=((1-laek14)**(jendie-y))*irabazixek14[y]*(laek14**y)*binomio
            laekue15+=((1-laek15)**(jendie-y))*irabazixek15[y]*(laek15**y)*binomio

        maxi = ((benetakue15 * laekue15) + (benetakue14 * laekue14) + (benetakue13[i] * laekue13) + (benetakue12[i] * laekue12) + (benetakue11[i] * laekue11)  + (benetakue10[i] * laekue10) )/0.75
        if maxi> 1.4:
            cont+=1
        bukaera.append([maxi,i,g])
bukaera = sorted(bukaera, key=lambda x: x[0], reverse=True)
bukaera = bukaera[:cont]
result(bukaera,konbodanak)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))