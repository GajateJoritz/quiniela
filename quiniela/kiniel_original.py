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
        print(resultaue[lista[i][1][0][0]],resultaue[lista[i][1][0][1]],resultaue[lista[i][1][0][2]],resultaue[lista[i][1][0][3]],resultaue[lista[i][1][0][4]],resultaue[lista[i][1][0][5]],resultaue[lista[i][1][0][6]],resultaue[lista[i][1][0][7]],resultaue[lista[i][1][0][8]],resultaue[lista[i][1][0][9]],resultaue[lista[i][1][0][10]],resultaue[lista[i][1][0][11]],resultaue[lista[i][1][0][12]],resultaue[lista[i][1][0][13]])
        print(plenue[lista[i][1][0][14]])
        f.write(resultaue[lista[i][1][0][0]] + resultaue[lista[i][1][0][1]] + resultaue[lista[i][1][0][2]] + resultaue[lista[i][1][0][3]] + resultaue[lista[i][1][0][4]] + resultaue[lista[i][1][0][5]] + resultaue[lista[i][1][0][6]] + resultaue[lista[i][1][0][7]] + resultaue[lista[i][1][0][8]] + resultaue[lista[i][1][0][9]] + resultaue[lista[i][1][0][10]] + resultaue[lista[i][1][0][11]] + resultaue[lista[i][1][0][12]] + resultaue[lista[i][1][0][13]] + plen[lista[i][1][0][14]] + "\n")
    f.close()

def portz(lista):
    total=0.0
    for i in range(len(lista)):
            total+=1/lista[i]
    for i in range(len(lista)):
        lista[i]=1/(total*lista[i])

def faltadinak(bat,bi):
    for z in range(0,13):    
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break

def faltadin(bat,bi):
    for z in range(0,12):
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break

bene=[]
lae=[]
"""
tree=ET.parse("lae.xml")
root=tree.getroot()
for elem in root:
    for part in elem:
        if part.get("num")!="15":
            laetxiki=[int((part.get("porc_1")))/100]
            laetxiki.append(int((part.get("porc_X")))/100)
            laetxiki.append(int((part.get("porc_2")))/100)
            lae.append(laetxiki)
            laetxiki=[]
        else:   
            pa1=(int((part.get("porc_15L_0")))/100)
            pa2=(int((part.get("porc_15L_1")))/100)
            pa3=(int((part.get("porc_15L_2")))/100)
            pa4=(int((part.get("porc_15L_M")))/100)
            pb1=(int((part.get("porc_15V_0")))/100)
            pb2=(int((part.get("porc_15V_1")))/100)
            pb3=(int((part.get("porc_15V_2")))/100)
            pb4=(int((part.get("porc_15V_M")))/100)
lae.append([pa1*pb1,pa1*pb2,pa1*pb3,pa1*pb4,pa2*pb1,pa2*pb2,pa2*pb3,pa2*pb4,pa3*pb1,pa3*pb2,pa3*pb3,pa3*pb4,pa4*pb1,pa4*pb2,pa4*pb3,pa4*pb4])

"""
with open("estimacion.txt", mode="r", encoding="utf-8-sig") as datos:
    valores=[]
    for linea in datos:
        linea=linea.replace(",",".")
        valores.append([float(x)/100 for x in linea.split()])
for i in range(0,14):
    lae.append([valores[i][0],valores[i][1],valores[i][2]])
for i in range(14,15):
    lae.append([valores[i][0]*valores[i+1][0],valores[i][0]*valores[i+1][1],valores[i][0]*valores[i+1][2],valores[i][0]*valores[i+1][3],valores[i][1]*valores[i+1][0],valores[i][1]*valores[i+1][1],valores[i][1]*valores[i+1][2],valores[i][1]*valores[i+1][3],valores[i][2]*valores[i+1][0],valores[i][2]*valores[i+1][1],valores[i][2]*valores[i+1][2],valores[i][2]*valores[i+1][3],valores[i][3]*valores[i+1][0],valores[i][3]*valores[i+1][1],valores[i][3]*valores[i+1][2],valores[i][3]*valores[i+1][3]])

tree=ET.parse("bene.xml")
root=tree.getroot()
for elem in root:
    for part in elem:
        if part.get("num")!="15":
            benetxiki=[float((part.get("porcDec_1")))/100]
            benetxiki.append(float((part.get("porcDec_X")))/100)
            benetxiki.append(float((part.get("porcDec_2")))/100)
            bene.append(benetxiki)
            benetxiki=[]
datos=[7,6.5,11,23,7,6,10,19,13,12,19,29,29,23,41,56]
portz(datos)
bene.append(datos)

#76.527.504 konbinazixo
estimazixue = 4266667
kantidadie=1
bukaera=[]
for i in range(1944):
    bukaera.append([0,[]])

irabazixek10=[0] * 1000
irabazixek11=[0] * 1000
irabazixek12=[0] * 1000
irabazixek13=[0] * 1000
irabazixek14=[0] * 1000
irabazixek15=[0] * 1000
konbodanak=[]

for yi in range(0,1000):
    estim=estimazixue*0.075*0.75
    if (estim+848000)/(yi+1)>40000:
        irabazixek15[yi]=((estim+848000)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek15[yi]=(estim+848000)/(yi+1)
    if (estimazixue*0.16*0.75)/(yi+1)>40000:
        irabazixek14[yi]=((estimazixue*0.16*0.75)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek14[yi]=(estimazixue*0.16*0.75)/(yi+1)
    
    if (estim)/(yi+1)>40000:
        irabazixek13[yi]=((estim)/(yi+1)-40000)*0.8+40000
        irabazixek12[yi]=((estim)/(yi+1)-40000)*0.8+40000
        irabazixek11[yi]=((estim)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek13[yi]=(estim)/(yi+1)
        irabazixek12[yi]=(estim)/(yi+1)
        irabazixek11[yi]=(estim)/(yi+1)
    if (estimazixue*0.09*0.75)/(yi+1)>40000:
        irabazixek10[yi]=((estimazixue*0.09*0.75)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek10[yi]=(estimazixue*0.09*0.75)/(yi+1)
  
konbodanak=np.load("motza.npy")
print(len(konbodanak))                                              
for x in combinations(konbodanak,kantidadie):
    hamabi=0.0
    hamahiru=0.0
    hamalau=0.0
    hamabost=0.0
    for xi in x:  
        benetakue14=(bene[0][xi[0]])*(bene[1][xi[1]])*(bene[2][xi[2]])*(bene[3][xi[3]])*(bene[4][xi[4]])*(bene[5][xi[5]])*(bene[6][xi[6]])*(bene[7][xi[7]])*(bene[8][xi[8]])*(bene[9][xi[9]])*(bene[10][xi[10]])*(bene[11][xi[11]])*(bene[12][xi[12]])*(bene[13][xi[13]])
        benetakue15= benetakue14 * (bene[14][xi[14]])
        laek14= (lae[0][xi[0]])*(lae[1][xi[1]])*(lae[2][xi[2]])*(lae[3][xi[3]])*(lae[4][xi[4]])*(lae[5][xi[5]])*(lae[6][xi[6]])*(lae[7][xi[7]])*(lae[8][xi[8]])*(lae[9][xi[9]])*(lae[10][xi[10]])*(lae[11][xi[11]])*(lae[12][xi[12]])*(lae[13][xi[13]])
        laek15 = laek14 * (lae[14][xi[14]])
        laekue15=0.0
        laekue14=0.0
        laekue13=0.0
        laekue12=0.0
        laek13=0.0
        benetakue13=0.0
        laek12=0.0
        benetakue12=0.0
        ben=[bene[0][xi[0]],bene[1][xi[1]],bene[2][xi[2]],bene[3][xi[3]],bene[4][xi[4]],bene[5][xi[5]],bene[6][xi[6]],bene[7][xi[7]],bene[8][xi[8]],bene[9][xi[9]],bene[10][xi[10]],bene[11][xi[11]],bene[12][xi[12]],bene[13][xi[13]]]
        la=[lae[0][xi[0]],lae[1][xi[1]],lae[2][xi[2]],lae[3][xi[3]],lae[4][xi[4]],lae[5][xi[5]],lae[6][xi[6]],lae[7][xi[7]],lae[8][xi[8]],lae[9][xi[9]],lae[10][xi[10]],lae[11][xi[11]],lae[12][xi[12]],lae[13][xi[13]]]
        konbin=(list(combinations([ben[0],ben[1],ben[2],ben[3],ben[4],ben[5],ben[6],ben[7],ben[8],ben[9],ben[10],ben[11],ben[12],ben[13]], 13)))
        konbin2=(list(combinations([la[0],la[1],la[2],la[3],la[4],la[5],la[6],la[7],la[8],la[9],la[10],la[11],la[12],la[13]], 13)))
        konbinazix=(list(combinations([ben[0],ben[1],ben[2],ben[3],ben[4],ben[5],ben[6],ben[7],ben[8],ben[9],ben[10],ben[11],ben[12],ben[13]], 12)))
        konbinazix2=(list(combinations([la[0],la[1],la[2],la[3],la[4],la[5],la[6],la[7],la[8],la[9],la[10],la[11],la[12],la[13]], 12)))
        for z in range(0,14):
            sobrak=[bene[0][xi[0]],bene[1][xi[1]],bene[2][xi[2]],bene[3][xi[3]],bene[4][xi[4]],bene[5][xi[5]],bene[6][xi[6]],bene[7][xi[7]],bene[8][xi[8]],bene[9][xi[9]],bene[10][xi[10]],bene[11][xi[11]],bene[12][xi[12]],bene[13][xi[13]]]
            sobrak2=[lae[0][xi[0]],lae[1][xi[1]],lae[2][xi[2]],lae[3][xi[3]],lae[4][xi[4]],lae[5][xi[5]],lae[6][xi[6]],lae[7][xi[7]],lae[8][xi[8]],lae[9][xi[9]],lae[10][xi[10]],lae[11][xi[11]],lae[12][xi[12]],lae[13][xi[13]]]
            faltadinak(sobrak,konbin[z])                                                       
            faltadinak(sobrak2,konbin2[z])
            benetakue13+= (konbin[z][0])*(konbin[z][1])*(konbin[z][2])*(konbin[z][3])*(konbin[z][4])*(konbin[z][5])*(konbin[z][6])*(konbin[z][7])*(konbin[z][8])*(konbin[z][9])*(konbin[z][10])*(konbin[z][11])*(konbin[z][12])*(1-sobrak[0])
            laek13+= (konbin2[z][0])*(konbin2[z][1])*(konbin2[z][2])*(konbin2[z][3])*(konbin2[z][4])*(konbin2[z][5])*(konbin2[z][6])*(konbin2[z][7])*(konbin2[z][8])*(konbin2[z][9])*(konbin2[z][10])*(konbin2[z][11])*(konbin2[z][12])*(1-sobrak2[0])
        #for zi in range(0,91):
            #sobrak3=[bene[0][xi[0]],bene[1][xi[1]],bene[2][xi[2]],bene[3][xi[3]],bene[4][xi[4]],bene[5][xi[5]],bene[6][xi[6]],bene[7][xi[7]],bene[8][xi[8]],bene[9][xi[9]],bene[10][xi[10]],bene[11][xi[11]],bene[12][xi[12]],bene[13][xi[13]]]
            #sobrak4=[lae[0][xi[0]],lae[1][xi[1]],lae[2][xi[2]],lae[3][xi[3]],lae[4][xi[4]],lae[5][xi[5]],lae[6][xi[6]],lae[7][xi[7]],lae[8][xi[8]],lae[9][xi[9]],lae[10][xi[10]],lae[11][xi[11]],lae[12][xi[12]],lae[13][xi[13]]]
            #faltadin(sobrak3,konbinazix[zi])
            #faltadin(sobrak4,konbinazix2[zi])
            #benetakue12+= (konbinazix[zi][0])*(konbinazix[zi][1])*(konbinazix[zi][2])*(konbinazix[zi][3])*(konbinazix[zi][4])*(konbinazix[zi][5])*(konbinazix[zi][6])*(konbinazix[zi][7])*(konbinazix[zi][8])*(konbinazix[zi][9])*(konbinazix[zi][10])*(konbinazix[zi][11])*(1-sobrak3[0])*(1-sobrak3[1])
            #laek12+= (konbinazix2[zi][0])*(konbinazix2[zi][1])*(konbinazix2[zi][2])*(konbinazix2[zi][3])*(konbinazix2[zi][4])*(konbinazix2[zi][5])*(konbinazix2[zi][6])*(konbinazix2[zi][7])*(konbinazix2[zi][8])*(konbinazix2[zi][9])*(konbinazix2[zi][10])*(konbinazix2[zi][11])*(1-sobrak4[0])*(1-sobrak4[1])
        for y in range(0,50):
            #estimazixue+1 arte berez
            binomio=scipy.special.comb(estimazixue, y, exact=True)
            #binomio=sympy.binomial(estimazixue,y)
            #binomio=binomial_mpmath(estimazixue,y)
            #laekue12+=((1-laek12)**(estimazixue-y))*irabazixek12[y]*(laek12**y)*binomio
            laekue13+=((1-laek13)**(estimazixue-y))*irabazixek13[y]*(laek13**y)*binomio
            laekue14+=((1-laek14)**(estimazixue-y))*irabazixek14[y]*(laek14**y)*binomio
            laekue15+=((1-laek15)**(estimazixue-y))*irabazixek15[y]*(laek15**y)*binomio
            
        hamabi+= benetakue12 * laekue12
        hamahiru+= benetakue13 * laekue13
        hamalau+= benetakue14 * laekue14
        hamabost+= benetakue15 * laekue15

    maxi = (hamabost+hamalau+hamahiru+hamabi)/(0.75*kantidadie)
    for i in range (0,len(bukaera)):
        if maxi>bukaera[i][0]:
            bukaera.insert(i,[maxi, x])
            del bukaera[len(bukaera)-1]
            break

result(bukaera)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))