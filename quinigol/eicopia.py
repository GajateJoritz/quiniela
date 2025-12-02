import time
import numpy as np
import scipy.special
from itertools import combinations
from datetime import timedelta
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
        #print(lista[i][0])
        #print(resultaue[konbodanak[lista[i][1]][0]],resultaue[konbodanak[lista[i][1]][1]],resultaue[konbodanak[lista[i][1]][2]],resultaue[konbodanak[lista[i][1]][3]],resultaue[konbodanak[lista[i][1]][4]],resultaue[konbodanak[lista[i][1]][5]])
        #print(lista[i][2])
        f.write(resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][1]] + resul[konbodanak[lista[i][1]][2]] + resul[konbodanak[lista[i][1]][3]] + resul[konbodanak[lista[i][1]][4]] + resul[konbodanak[lista[i][1]][5]] + "\n")
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

bene=[]
lae=[]
with open("laequinig.txt", mode="r") as datos:
    valores=[]
    for linea in datos:
        linea=linea.replace(",",".")
        valores.append([float(x)/100 for x in linea.strip().split(";")])
for i in range(0,12,2):
    lae.append([valores[i][0]*valores[i+1][0],valores[i][0]*valores[i+1][1],valores[i][0]*valores[i+1][2],valores[i][0]*valores[i+1][3],valores[i][1]*valores[i+1][0],valores[i][1]*valores[i+1][1],valores[i][1]*valores[i+1][2],valores[i][1]*valores[i+1][3],valores[i][2]*valores[i+1][0],valores[i][2]*valores[i+1][1],valores[i][2]*valores[i+1][2],valores[i][2]*valores[i+1][3],valores[i][3]*valores[i+1][0],valores[i][3]*valores[i+1][1],valores[i][3]*valores[i+1][2],valores[i][3]*valores[i+1][3]])


botie = 15000
estimazixue = 75000
bene1=[19,9.5,8,4.75,21,10,9,5.5,41,19,17,11,91,51,36,19]
bene2=[13,7.5,7.5,6,15,8,8.5,7,34,19,17,15,96,56,46,34]
bene3=[8,15,41,161,5.5,8,23,81,5.5,9,26,76,5.5,9,26,67]
bene4=[8,11,23,71,5.5,7,17,51,7,9,21,51,9,11,23,56]
bene5=[13,9,10,15,11,7,9,10,17,11,13,15,23,17,21,29]
bene6=[29,41,51,201,13,15,29,61,10,9.5,19,46,4,3.75,7.5,15]
portz(bene1)
portz(bene2)
portz(bene3)
portz(bene4)
portz(bene5)
portz(bene6)


#bene1=[0.07,0.098,0.0672,0.0448,0.0875,0.1225,0.084,0.056,0.0575,0.0805,0.0552,0.0368,0.035,0.049,0.036,0.0224]
#bene2=[0.0371,0.1007,0.1325,0.2597,0.0231,0.0627,0.0825,0.1617,0.0077,0.0209,0.0275,0.0539,0.0021,0.0057,0.0075,0.0147]
#bene3=[0.0416,0.1092,0.1352,0.234,0.0272,0.0714,0.0884,0.153,0.0088,0.0231,0.0286,0.0495,0.0024,0.0063,0.0078,0.0135]
#bene4=[0.0666,0.1147,0.0962,0.0925,0.0666,0.1147,0.0962,0.0925,0.0324,0.0558,0.0468,0.045,0.0144,0.0248,0.0208,0.02]
#bene5=[0.0646,0.1088,0.0884,0.0782,0.0703,0.1184,0.0962,0.0851,0.038,0.064,0.052,0.046,0.0171,0.0288,0.0234,0.0207]
#bene6=[0.0495,0.1125,0.1215,0.1665,0.0396,0.09,0.0972,0.1332,0.0154,0.035,0.0378,0.0518,0.0055,0.0125,0.0135,0.0185]

#16.777.216 konbinazixo
bene.append(bene1)
bene.append(bene2)
bene.append(bene3)
bene.append(bene4)
bene.append(bene5)
bene.append(bene6)


bukaera=[]

irabazixek2=[0] * 1000
irabazixek3=[0] * 1000
irabazixek4=[0] * 1000
irabazixek5=[0] * 1000
irabazixek6=[0] * 1000
konbodanak=[]

sobrantie = botie - estimazixue*0.1
if sobrantie < 0:
    sobrantie=0
for yi in range(0,1000):
    if (estimazixue*0.1+sobrantie)/(yi+1)>40000:
        irabazixek6[yi]=((estimazixue*0.1+sobrantie)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek6[yi]=(estimazixue*0.1+sobrantie)/(yi+1)
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

cont=0
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
    benetakue5=0.0
    laek5=0.0
    benetakue4=0.0
    laek4=0.0
    benetakue3=0.0
    laek3=0.0
    benetakue2=0.0
    laek2=0.0
    for zu in range(0,6):
        benetakue5+= (danakb[i][konbin[zu][0]])*(danakb[i][konbin[zu][1]])*(danakb[i][konbin[zu][2]])*(danakb[i][konbin[zu][3]])*(danakb[i][konbin[zu][4]])*(1-danakb[i][sobrkx[zu]])
        laek5+= (danakl[i][konbin[zu][0]])*(danakl[i][konbin[zu][1]])*(danakl[i][konbin[zu][2]])*(danakl[i][konbin[zu][3]])*(danakl[i][konbin[zu][4]])*(1-danakl[i][sobrkx[zu]])
    for z in range(0,15):
        benetakue4+= (danakb[i][konbinazixuk[z][0]])*(danakb[i][konbinazixuk[z][1]])*(danakb[i][konbinazixuk[z][2]])*(danakb[i][konbinazixuk[z][3]])*(1-danakb[i][sobrx[z][0]])*(1-danakb[i][sobrx[z][1]])
        laek4+= (danakl[i][konbinazixuk[z][0]])*(danakl[i][konbinazixuk[z][1]])*(danakl[i][konbinazixuk[z][2]])*(danakl[i][konbinazixuk[z][3]])*(1-danakl[i][sobrx[z][0]])*(1-danakl[i][sobrx[z][1]])
    for zi in range(0,20):
        benetakue3+= (danakb[i][konbinazix[zi][0]])*(danakb[i][konbinazix[zi][1]])*(danakb[i][konbinazix[zi][2]])*(1-danakb[i][sobx[zi][0]])*(1-danakb[i][sobx[zi][1]])*(1-danakb[i][sobx[zi][2]])
        laek3+= (danakl[i][konbinazix[zi][0]])*(danakl[i][konbinazix[zi][1]])*(danakl[i][konbinazix[zi][2]])*(1-danakl[i][sobx[zi][0]])*(1-danakl[i][sobx[zi][1]])*(1-danakl[i][sobx[zi][2]])
    for zo in range(0,15):
        benetakue2+= (danakb[i][konbinaz[zo][0]])*(danakb[i][konbinaz[zo][1]])*(1-danakb[i][sox[zo][0]])*(1-danakb[i][sox[zo][1]])*(1-danakb[i][sox[zo][2]])*(1-danakb[i][sox[zo][3]])
        laek2+= (danakl[i][konbinaz[zo][0]])*(danakl[i][konbinaz[zo][1]])*(1-danakl[i][sox[zo][0]])*(1-danakl[i][sox[zo][1]])*(1-danakl[i][sox[zo][2]])*(1-danakl[i][sox[zo][3]])
    for y in range(0,50):
        binomio=scipy.special.comb(estimazixue, y, exact=True)
        laekue6+= ((1-laek6)**(estimazixue-y))*irabazixek6[y]*(laek6**y)*binomio                   
        laekue5+= ((1-laek5)**(estimazixue-y))*irabazixek5[y]*(laek5**y)*binomio
        laekue4+= ((1-laek4)**(estimazixue-y))*irabazixek4[y]*(laek4**y)*binomio
        laekue3+= ((1-laek3)**(estimazixue-y))*irabazixek3[y]*(laek3**y)*binomio
        laekue2+= ((1-laek2)**(estimazixue-y))*irabazixek2[y]*(laek2**y)*binomio
    maxi= (benetakue6 * laekue6) + (benetakue5 * laekue5) + (benetakue4 * laekue4) + (benetakue3 * laekue3) + (benetakue2 * laekue2)
    if maxi> 1.4:
        cont+=1
        bukaera.append([maxi, i, benetakue6])
bukaera = sorted(bukaera, key=lambda x: x[2], reverse=True)
result(bukaera)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))    