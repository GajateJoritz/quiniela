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
import gmpy2

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

def result(lista):
    resultaue=["1","X","2"]
    plenue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    plen=["00","01","02","0M","10","11","12","1M","20","21","22","2M","M0","M1","M2","MM"]
    f= open("resul.txt", "w")
    for i in range(len(lista)):
        print(lista[i][0])
        print(resultaue[lista[i][1][0]],resultaue[lista[i][1][1]],resultaue[lista[i][1][2]],resultaue[lista[i][1][3]],resultaue[lista[i][1][4]],resultaue[lista[i][1][5]],resultaue[lista[i][1][6]],resultaue[lista[i][1][7]],resultaue[lista[i][1][8]],resultaue[lista[i][1][9]],resultaue[lista[i][1][10]],resultaue[lista[i][1][11]],resultaue[lista[i][1][12]],resultaue[lista[i][1][13]])
        print(plenue[lista[i][1][14]])

        f.write(resultaue[lista[i][1][0]] + resultaue[lista[i][1][1]] + resultaue[lista[i][1][2]] + resultaue[lista[i][1][3]] + resultaue[lista[i][1][4]] + resultaue[lista[i][1][5]] + resultaue[lista[i][1][6]] + resultaue[lista[i][1][7]] + resultaue[lista[i][1][8]] + resultaue[lista[i][1][9]] + resultaue[lista[i][1][10]] + resultaue[lista[i][1][11]] + resultaue[lista[i][1][12]] + resultaue[lista[i][1][13]] + plen[lista[i][1][14]] + "\n")
    f.close()

def portz(lista):
    total=0.0
    for i in range(len(lista)):
            total+=1/lista[i]
    for i in range(len(lista)):
        lista[i]=1/(total*lista[i])

bene=[]
lae=[]

with open("estimacion.txt", mode="r", encoding="utf-8-sig") as datos:
    valores=[]
    for linea in datos:
        linea=linea.replace(",",".")
        valores.append([float(x)/100 for x in linea.split()])
for i in range(0,14):
    lae.append([valores[i][0],valores[i][1],valores[i][2]])
for i in range(14,15):
    lae.append([valores[i][0]*valores[i+1][0],valores[i][0]*valores[i+1][1],valores[i][0]*valores[i+1][2],valores[i][0]*valores[i+1][3],valores[i][1]*valores[i+1][0],valores[i][1]*valores[i+1][1],valores[i][1]*valores[i+1][2],valores[i][1]*valores[i+1][3],valores[i][2]*valores[i+1][0],valores[i][2]*valores[i+1][1],valores[i][2]*valores[i+1][2],valores[i][2]*valores[i+1][3],valores[i][3]*valores[i+1][0],valores[i][3]*valores[i+1][1],valores[i][3]*valores[i+1][2],valores[i][3]*valores[i+1][3]])

url = "https://api.eduardolosilla.es/servicios/v1/probabilidad_real?jornada=19&temporada=2025"
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
datos=[7,9.5,21,56,5.5,6,15,41,8,9.5,19,46,12,13,23,61]
portz(datos)
bene.append(datos)

#76.527.504 konbinazixo
estimazixue = 3500000
jendie= estimazixue/0.75
bukaera=[]
for i in range(3888):
    bukaera.append([0,[]])

irabazixek10=[0] * 1000
irabazixek11=[0] * 1000
irabazixek12=[0] * 1000
irabazixek13=[0] * 1000
irabazixek14=[0] * 1000
irabazixek15=[0] * 1000
konbodanak=[]

for yi in range(0,1000):
    estim=estimazixue*0.075
    if (estim+1777500)/(yi+1)>40000:
        irabazixek15[yi]=((estim+1777500)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek15[yi]=(estim+1777500)/(yi+1)
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
  
konbodanak=np.load("motza.npy")
print(len(konbodanak))     

dan=[bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]]
dan2=[lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]]

numeros=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
sobrakx=[13,12,11,10,9,8,7,6,5,4,3,2,1,0]
kon = list(combinations(numeros, 13))
konb = list(combinations(numeros, 12))
ko = list(combinations(numeros, 11))
k = list(combinations(numeros, 10))
num = set(numeros)
sobrx=[]
sobx=[]
sox=[]
for elem in konb:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobrx.append(faltantes)
for elem in ko:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobx.append(faltantes)
for elem in k:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sox.append(faltantes) 
konbin = list(combinations(dan, 13))
konbin2 = list(combinations(dan2, 13))
konbinazix = list(combinations(dan, 12))
konbinazix2 = list(combinations(dan2, 12))
konbinazixuk = list(combinations(dan, 11))
konbinazixuk2 = list(combinations(dan2, 11))
kbinazixuk = list(combinations(dan, 10))
kbinazixuk2 = list(combinations(dan2, 10))
for xi in konbodanak:
    benetakue14=(bene[0][xi[0]])*(bene[1][xi[1]])*(bene[2][xi[2]])*(bene[3][xi[3]])*(bene[4][xi[4]])*(bene[5][xi[5]])*(bene[6][xi[6]])*(bene[7][xi[7]])*(bene[8][xi[8]])*(bene[9][xi[9]])*(bene[10][xi[10]])*(bene[11][xi[11]])*(bene[12][xi[12]])*(bene[13][xi[13]])
    benetakue15= benetakue14 * (bene[14][xi[14]])
    laek14= (lae[0][xi[0]])*(lae[1][xi[1]])*(lae[2][xi[2]])*(lae[3][xi[3]])*(lae[4][xi[4]])*(lae[5][xi[5]])*(lae[6][xi[6]])*(lae[7][xi[7]])*(lae[8][xi[8]])*(lae[9][xi[9]])*(lae[10][xi[10]])*(lae[11][xi[11]])*(lae[12][xi[12]])*(lae[13][xi[13]])
    laek15 = laek14 * (lae[14][xi[14]])
    laekue15=0.0
    laekue14=0.0
    laekue13=0.0
    laekue12=0.0
    laekue11=0.0
    laekue10=0.0
    laek13=0.0
    laek12=0.0
    laek11=0.0
    #laek10=0.0
    benetakue13=0.0
    benetakue12=0.0
    benetakue11=0.0
    benetakue10=0.0

    for z in range(0,14):                                               
        benetakue13+= (konbin[z][0][xi[kon[z][0]]])*(konbin[z][1][xi[kon[z][1]]])*(konbin[z][2][xi[kon[z][2]]])*(konbin[z][3][xi[kon[z][3]]])*(konbin[z][4][xi[kon[z][4]]])*(konbin[z][5][xi[kon[z][5]]])*(konbin[z][6][xi[kon[z][6]]])*(konbin[z][7][xi[kon[z][7]]])*(konbin[z][8][xi[kon[z][8]]])*(konbin[z][9][xi[kon[z][9]]])*(konbin[z][10][xi[kon[z][10]]])*(konbin[z][11][xi[kon[z][11]]])*(konbin[z][12][xi[kon[z][12]]])*(1-dan[sobrakx[z]][xi[sobrakx[z]]])
        laek13+= (konbin2[z][0][xi[kon[z][0]]])*(konbin2[z][1][xi[kon[z][1]]])*(konbin2[z][2][xi[kon[z][2]]])*(konbin2[z][3][xi[kon[z][3]]])*(konbin2[z][4][xi[kon[z][4]]])*(konbin2[z][5][xi[kon[z][5]]])*(konbin2[z][6][xi[kon[z][6]]])*(konbin2[z][7][xi[kon[z][7]]])*(konbin2[z][8][xi[kon[z][8]]])*(konbin2[z][9][xi[kon[z][9]]])*(konbin2[z][10][xi[kon[z][10]]])*(konbin2[z][11][xi[kon[z][11]]])*(konbin2[z][12][xi[kon[z][12]]])*(1-dan2[sobrakx[z]][xi[sobrakx[z]]])
    for zi in range(0,91):
        benetakue12+= (konbinazix[zi][0][xi[konb[zi][0]]])*(konbinazix[zi][1][xi[konb[zi][1]]])*(konbinazix[zi][2][xi[konb[zi][2]]])*(konbinazix[zi][3][xi[konb[zi][3]]])*(konbinazix[zi][4][xi[konb[zi][4]]])*(konbinazix[zi][5][xi[konb[zi][5]]])*(konbinazix[zi][6][xi[konb[zi][6]]])*(konbinazix[zi][7][xi[konb[zi][7]]])*(konbinazix[zi][8][xi[konb[zi][8]]])*(konbinazix[zi][9][xi[konb[zi][9]]])*(konbinazix[zi][10][xi[konb[zi][10]]])*(konbinazix[zi][11][xi[konb[zi][11]]])*(1-dan[sobrx[zi][0]][xi[sobrx[zi][0]]])*(1-dan[sobrx[zi][1]][xi[sobrx[zi][1]]])
        laek12+= (konbinazix2[zi][0][xi[konb[zi][0]]])*(konbinazix2[zi][1][xi[konb[zi][1]]])*(konbinazix2[zi][2][xi[konb[zi][2]]])*(konbinazix2[zi][3][xi[konb[zi][3]]])*(konbinazix2[zi][4][xi[konb[zi][4]]])*(konbinazix2[zi][5][xi[konb[zi][5]]])*(konbinazix2[zi][6][xi[konb[zi][6]]])*(konbinazix2[zi][7][xi[konb[zi][7]]])*(konbinazix2[zi][8][xi[konb[zi][8]]])*(konbinazix2[zi][9][xi[konb[zi][9]]])*(konbinazix2[zi][10][xi[konb[zi][10]]])*(konbinazix2[zi][11][xi[konb[zi][11]]])*(1-dan2[sobrx[zi][0]][xi[sobrx[zi][0]]])*(1-dan2[sobrx[zi][1]][xi[sobrx[zi][1]]])
    for zo in range(0,364):
        benetakue11+= (konbinazixuk[zo][0][xi[ko[zo][0]]])*(konbinazixuk[zo][1][xi[ko[zo][1]]])*(konbinazixuk[zo][2][xi[ko[zo][2]]])*(konbinazixuk[zo][3][xi[ko[zo][3]]])*(konbinazixuk[zo][4][xi[ko[zo][4]]])*(konbinazixuk[zo][5][xi[ko[zo][5]]])*(konbinazixuk[zo][6][xi[ko[zo][6]]])*(konbinazixuk[zo][7][xi[ko[zo][7]]])*(konbinazixuk[zo][8][xi[ko[zo][8]]])*(konbinazixuk[zo][9][xi[ko[zo][9]]])*(konbinazixuk[zo][10][xi[ko[zo][10]]])*(1-dan[sobx[zo][0]][xi[sobx[zo][0]]])*(1-dan[sobx[zo][1]][xi[sobx[zo][1]]])*(1-dan[sobx[zo][2]][xi[sobx[zo][2]]])
        laek11+= (konbinazixuk2[zo][0][xi[ko[zo][0]]])*(konbinazixuk2[zo][1][xi[ko[zo][1]]])*(konbinazixuk2[zo][2][xi[ko[zo][2]]])*(konbinazixuk2[zo][3][xi[ko[zo][3]]])*(konbinazixuk2[zo][4][xi[ko[zo][4]]])*(konbinazixuk2[zo][5][xi[ko[zo][5]]])*(konbinazixuk2[zo][6][xi[ko[zo][6]]])*(konbinazixuk2[zo][7][xi[ko[zo][7]]])*(konbinazixuk2[zo][8][xi[ko[zo][8]]])*(konbinazixuk2[zo][9][xi[ko[zo][9]]])*(konbinazixuk2[zo][10][xi[ko[zo][10]]])*(1-dan2[sobx[zo][0]][xi[sobx[zo][0]]])*(1-dan2[sobx[zo][1]][xi[sobx[zo][1]]])*(1-dan2[sobx[zo][2]][xi[sobx[zo][2]]])
    #for zu in range(0,1001):
    #    benetakue10+= (kbinazixuk[zu][0][xi[k[zu][0]]])*(kbinazixuk[zu][1][xi[k[zu][1]]])*(kbinazixuk[zu][2][xi[k[zu][2]]])*(kbinazixuk[zu][3][xi[k[zu][3]]])*(kbinazixuk[zu][4][xi[k[zu][4]]])*(kbinazixuk[zu][5][xi[k[zu][5]]])*(kbinazixuk[zu][6][xi[k[zu][6]]])*(kbinazixuk[zu][7][xi[k[zu][7]]])*(kbinazixuk[zu][8][xi[k[zu][8]]])*(kbinazixuk[zu][9][xi[k[zu][9]]])*(1-dan[sox[zu][0]][xi[sox[zu][0]]])*(1-dan[sox[zu][1]][xi[sox[zu][1]]])*(1-dan[sox[zu][2]][xi[sox[zu][2]]])*(1-dan[sox[zu][3]][xi[sox[zu][3]]])
    #    laek10+= (kbinazixuk2[zu][0][xi[k[zu][0]]])*(kbinazixuk2[zu][1][xi[k[zu][1]]])*(kbinazixuk2[zu][2][xi[k[zu][2]]])*(kbinazixuk2[zu][3][xi[k[zu][3]]])*(kbinazixuk2[zu][4][xi[k[zu][4]]])*(kbinazixuk2[zu][5][xi[k[zu][5]]])*(kbinazixuk2[zu][6][xi[k[zu][6]]])*(kbinazixuk2[zu][7][xi[k[zu][7]]])*(kbinazixuk2[zu][8][xi[k[zu][8]]])*(kbinazixuk2[zu][9][xi[k[zu][9]]])*(1-dan2[sox[zu][0]][xi[sox[zu][0]]])*(1-dan2[sox[zu][1]][xi[sox[zu][1]]])*(1-dan2[sox[zu][2]][xi[sox[zu][2]]])*(1-dan2[sox[zu][3]][xi[sox[zu][3]]])
    for y in range(0,50):
        #jendie+1 arte berez
        binomio=scipy.special.comb(jendie, y, exact=True)
        #binomio=gmpy2.comb(jendie,y)
        #binomio=sympy.binomial(jendie,y)
        #binomio=binomial_mpmath(jendie,y)
        #laekue10+=((1-laek10)**(jendie-y))*irabazixek10[y]*(laek10**y)*binomio
        laekue11+=((1-laek11)**(jendie-y))*irabazixek11[y]*(laek11**y)*binomio
        laekue12+=((1-laek12)**(jendie-y))*irabazixek12[y]*(laek12**y)*binomio
        laekue13+=((1-laek13)**(jendie-y))*irabazixek13[y]*(laek13**y)*binomio
        laekue14+=((1-laek14)**(jendie-y))*irabazixek14[y]*(laek14**y)*binomio
        laekue15+=((1-laek15)**(jendie-y))*irabazixek15[y]*(laek15**y)*binomio

    maxi = ((benetakue15 * laekue15) + (benetakue14 * laekue14) + (benetakue13 * laekue13) + (benetakue12 * laekue12) + (benetakue11 * laekue11)  + (benetakue10 * laekue10) )/0.75
    for i in range (0,len(bukaera)):
        if maxi>bukaera[i][0]:
            bukaera.insert(i,[maxi, xi])
            del bukaera[len(bukaera)-1]
            break
result(bukaera)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))