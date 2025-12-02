import time
import numpy as np
import scipy.special
from itertools import combinations
from datetime import timedelta
from numba import njit, prange
start_time = time.monotonic()

def diferent5(lista):
    konpa=[[0,0,0,0,2,0]]
    for e in konpa:
        total=0
        for i in range(0,6):
                if lista[i]==e[i]:
                    total+=1
        if total>=5:
            return True

def result(lista):
    resultaue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    resul=["00","01","02","0M","10","11","12","1M","20","21","22","2M","M0","M1","M2","MM"]
    f= open("resulQuinigol.txt", "w")
    for i in range(len(lista)):
        print(lista[i][0])
        print(resultaue[konbodanak[lista[i][1]][0]],resultaue[konbodanak[lista[i][1]][1]],resultaue[konbodanak[lista[i][1]][2]],resultaue[konbodanak[lista[i][1]][3]],resultaue[konbodanak[lista[i][1]][4]],resultaue[konbodanak[lista[i][1]][5]])
        print(lista[i][2])
        f.write(resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][0]] + "\n")
    f.close()

def portz(lista):
    f= open("portzentajeRealaKinigol.txt", "w")
    total=0.0
    for i in range(len(lista)):
            total+=1/lista[i]
    for i in range(len(lista)):
        lista[i]=1/(total*lista[i])
        f.write(str(lista[i]) + " ")
    f.write("\n")
    f.close()

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
with open("laequinig.txt", mode="r") as datos:
    valores=[]
    for linea in datos:
        linea=linea.replace(",",".")
        valores.append([float(x)/100 for x in linea.strip().split(";")])
for i in range(0,12,2):
    lae.append([valores[i][0]*valores[i+1][0],valores[i][0]*valores[i+1][1],valores[i][0]*valores[i+1][2],valores[i][0]*valores[i+1][3],valores[i][1]*valores[i+1][0],valores[i][1]*valores[i+1][1],valores[i][1]*valores[i+1][2],valores[i][1]*valores[i+1][3],valores[i][2]*valores[i+1][0],valores[i][2]*valores[i+1][1],valores[i][2]*valores[i+1][2],valores[i][2]*valores[i+1][3],valores[i][3]*valores[i+1][0],valores[i][3]*valores[i+1][1],valores[i][3]*valores[i+1][2],valores[i][3]*valores[i+1][3]])

bene1=[12,8.5,10,11,12,7,8.5,9.5,21,13,13,13,36,23,26,23]
bene2=[7.5,9,17,41,6,6,13,29,9,9.5,17,36,15,15,26,51]
bene3=[23,19,23,34,13,9,13,17,13,9,11,15,10,7,9,10]
bene4=[5,7.5,17,46,5,6,17,46,8.5,11,23,61,17,19,41,81]
bene5=[8.5,7,10,17,8,6.5,9.5,15,15,11,17,23,34,23,29,41]
bene6=[8,11,23,71,6,7,17,46,7.5,9,19,46,9.5,11,21,46]
portz(bene1)
portz(bene2)
portz(bene3)
portz(bene4)
portz(bene5)
portz(bene6)

#16.777.216 konbinazixo
bene.append(bene1)
bene.append(bene2)
bene.append(bene3)
bene.append(bene4)
bene.append(bene5)
bene.append(bene6)

estimazixue = 100000
bukaera=[]
for i in range(81):
    bukaera.append([0,[]])

irabazixek2=[0] * 1000
irabazixek3=[0] * 1000
irabazixek4=[0] * 1000
irabazixek5=[0] * 1000
irabazixek6=[0] * 1000
konbodanak=[]
kantidadeOnak=0

for yi in range(0,1000):
    if (estimazixue*0.1+60000)/(yi+1)>40000:
        irabazixek6[yi]=((estimazixue*0.1+60000)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek6[yi]=(estimazixue*0.1+60000)/(yi+1)
    if (estimazixue*0.09)/(yi+1)>40000:
        irabazixek5[yi]=((estimazixue*0.09)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek5[yi]=(estimazixue*0.09)/(yi+1)
    if (estimazixue*0.08)/(yi+1)>40000:
        irabazixek4[yi]=((estimazixue*0.08)/(yi+1)-40000)*0.8+40000
        irabazixek3[yi]=((estimazixue*0.08)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek4[yi]=(estimazixue*0.08)/(yi+1)
        irabazixek3[yi]=(estimazixue*0.08)/(yi+1)
    if (estimazixue*0.2)/(yi+1)>40000:
        irabazixek2[yi]=((estimazixue*0.2)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek2[yi]=(estimazixue*0.2)/(yi+1)
    
konbodanak=np.load("kinigmotza.npy")
print(len(konbodanak))  

numeros=[0,1,2,3,4,5]
sobrkx=[5,4,3,2,1,0]
konbin = list(combinations(numeros, 5))
konbinazixuk = list(combinations(numeros, 4))
konbinazix = list(combinations(numeros, 3))
konbinaz = list(combinations(numeros, 2))
num = set(numeros)
sobrx=[]
sobx=[]
sox=[]
for elem in konbinazixuk:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobrx.append(faltantes) 
for elem in konbinazix:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobx.append(faltantes)
for elem in konbinaz:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sox.append(faltantes)
     
danakb = []
danakl = []     
for xi in konbodanak:
    sub_b = []
    sub_l = []
    for i in range(0,6):
        sub_b.append(bene[i][xi[i]])
        sub_l.append(lae[i][xi[i]])
    danakb.append(sub_b)
    danakl.append(sub_l)

benetakue5 = calcular_benetakue(np.array(danakb), np.array(konbin), np.array(sobrkx))
benetakue4 = calcular_benetakue2(np.array(danakb), np.array(konbinazixuk), np.array(sobrx))
benetakue3 = calcular_benetakue2(np.array(danakb), np.array(konbinazix), np.array(sobx))
benetakue2 = calcular_benetakue2(np.array(danakb), np.array(konbinaz), np.array(sox))
laek5 = calcular_benetakue(np.array(danakl), np.array(konbin), np.array(sobrkx))
laek4 = calcular_benetakue2(np.array(danakl), np.array(konbinazixuk), np.array(sobrx))
laek3 = calcular_benetakue2(np.array(danakl), np.array(konbinazix), np.array(sobx))
laek2= calcular_benetakue2(np.array(danakl), np.array(konbinaz), np.array(sox))

for i in range(len(danakb)):
    #if diferent5(xi):
    #    continue
    benetakue6=(danakb[i][0])*(danakb[i][1])*(danakb[i][2])*(danakb[i][3])*(danakb[i][4])*(danakb[i][5])
    #if benetakue6 < 0.0000001:
    #    continue
    laek6=(danakl[i][0])*(danakl[i][1])*(danakl[i][2])*(danakl[i][3])*(danakl[i][4])*(danakl[i][5])
    laekue6=0.0
    laekue5=0.0
    laekue4=0.0
    laekue3=0.0
    laekue2=0.0
    for y in range(0,50):
        binomio=scipy.special.comb(estimazixue, y, exact=True)
        laekue6+= ((1-laek6)**(estimazixue-y))*irabazixek6[y]*(laek6**y)*binomio                   
        laekue5+= ((1-laek5[i])**(estimazixue-y))*irabazixek5[y]*(laek5[i]**y)*binomio
        laekue4+= ((1-laek4[i])**(estimazixue-y))*irabazixek4[y]*(laek4[i]**y)*binomio
        laekue3+= ((1-laek3[i])**(estimazixue-y))*irabazixek3[y]*(laek3[i]**y)*binomio
        laekue2+= ((1-laek2[i])**(estimazixue-y))*irabazixek2[y]*(laek2[i]**y)*binomio
    maxi= (benetakue6 * laekue6) + (benetakue5[i] * laekue5) + (benetakue4[i] * laekue4) + (benetakue3[i] * laekue3) + (benetakue2[i] * laekue2)
    #if maxi>1:
    #    kantidadeOnak+=1
    for j in range (0,len(bukaera)):
        if maxi>bukaera[j][0]:
            bukaera.insert(j,[maxi, i, benetakue6])
            del bukaera[len(bukaera)-1]
            break

result(bukaera)
#print("Onak: ", kantidadeOnak)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))    