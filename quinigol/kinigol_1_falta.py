import time
import numpy as np
import scipy.special
from itertools import combinations
from datetime import timedelta
start_time = time.monotonic()

def result(lista):
    resultaue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    resul=["00","01","02","0M","10","11","12","1M","20","21","22","2M","M0","M1","M2","MM"]
    f= open("resul_kinigol_1_falta.txt", "w")
    for i in range(len(lista)):
        print(lista[i][0])
        print(resultaue[konbodanak[lista[i][1]][0]],resultaue[konbodanak[lista[i][1]][1]],resultaue[konbodanak[lista[i][1]][2]],resultaue[konbodanak[lista[i][1]][3]],resultaue[konbodanak[lista[i][1]][4]])
        print(lista[i][2])
        f.write(resul[konbodanak[lista[i][1]][0]] + resul[konbodanak[lista[i][1]][1]] + resul[konbodanak[lista[i][1]][2]] + resul[konbodanak[lista[i][1]][3]] + resul[konbodanak[lista[i][1]][4]] + "\n")
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
for i in range(0,10,2):
    lae.append([valores[i][0]*valores[i+1][0],valores[i][0]*valores[i+1][1],valores[i][0]*valores[i+1][2],valores[i][0]*valores[i+1][3],valores[i][1]*valores[i+1][0],valores[i][1]*valores[i+1][1],valores[i][1]*valores[i+1][2],valores[i][1]*valores[i+1][3],valores[i][2]*valores[i+1][0],valores[i][2]*valores[i+1][1],valores[i][2]*valores[i+1][2],valores[i][2]*valores[i+1][3],valores[i][3]*valores[i+1][0],valores[i][3]*valores[i+1][1],valores[i][3]*valores[i+1][2],valores[i][3]*valores[i+1][3]])

bene1=[19,29,51,201,9.5,11,29,81,7.5,9,21,61,3.75,4.5,10,26]
bene2=[6.5,6,9.5,15,7.5,6,10,17,15,13,19,29,41,34,46,61]
bene3=[6,6.5,12,26,6.5,6,11,23,12,12,21,36,26,26,36,67]
bene4=[8,6,7.5,9.5,11,6.5,8.5,11,23,15,17,21,67,41,51,46]
bene5=[15,9,9.5,8.5,15,8,8.5,7.5,23,15,15,12,46,29,23,19]
portz(bene1)
portz(bene2)
portz(bene3)
portz(bene4)
portz(bene5)

#16.777.216 konbinazixo
bene.append(bene1)
bene.append(bene2)
bene.append(bene3)
bene.append(bene4)
bene.append(bene5)

estimazixue = 100000
bukaera=[]
for i in range(162):
    bukaera.append([0,[]])

irabazixek2=[0] * 1000
irabazixek3=[0] * 1000
irabazixek4=[0] * 1000
irabazixek5=[0] * 1000
konbodanak=[]
kantidadeOnak=0

for yi in range(0,1000):
    if (estimazixue*0.14+1000)/(yi+1)>40000:
        irabazixek5[yi]=((estimazixue*0.14+1000)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek5[yi]=(estimazixue*0.14+1000)/(yi+1)
    if (estimazixue*0.09)/(yi+1)>40000:
        irabazixek4[yi]=((estimazixue*0.09)/(yi+1)-40000)*0.8+40000
        irabazixek3[yi]=((estimazixue*0.09)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek4[yi]=(estimazixue*0.09)/(yi+1)
        irabazixek3[yi]=(estimazixue*0.09)/(yi+1)
    if (estimazixue*0.23)/(yi+1)>40000:
        irabazixek2[yi]=((estimazixue*0.23)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek2[yi]=(estimazixue*0.23)/(yi+1)
    
konbodanak=np.load("1_falta_kinigol_konbuk.npy")
print(len(konbodanak))  

numeros=[0,1,2,3,4]
konbinazixuk = list(combinations(numeros, 4))
konbinazix = list(combinations(numeros, 3))
konbinaz = list(combinations(numeros, 2))
num = set(numeros)
sobrx=[4,3,2,1,0]
sobx=[]
sox=[]
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
    for i in range(0,5):
        sub_b.append(bene[i][xi[i]])
        sub_l.append(lae[i][xi[i]])
    danakb.append(sub_b)
    danakl.append(sub_l)

for i in range(len(danakb)):
    benetakue5=(danakb[i][0])*(danakb[i][1])*(danakb[i][2])*(danakb[i][3])*(danakb[i][4])
    laek5=(danakl[i][0])*(danakl[i][1])*(danakl[i][2])*(danakl[i][3])*(danakl[i][4])
    laekue5=0.0
    laekue4=0.0
    laekue3=0.0
    laekue2=0.0
    laek5=0.0
    benetakue4=0.0
    laek4=0.0
    benetakue3=0.0
    laek3=0.0
    benetakue2=0.0
    laek2=0.0
    for z in range(0,5):
        benetakue4+= (danakb[i][konbinazixuk[z][0]])*(danakb[i][konbinazixuk[z][1]])*(danakb[i][konbinazixuk[z][2]])*(danakb[i][konbinazixuk[z][3]])*(1-danakb[i][sobrx[z]])
        laek4+= (danakl[i][konbinazixuk[z][0]])*(danakl[i][konbinazixuk[z][1]])*(danakl[i][konbinazixuk[z][2]])*(danakl[i][konbinazixuk[z][3]])*(1-danakl[i][sobrx[z]])
    for zi in range(0,10):
        benetakue3+= (danakb[i][konbinazix[zi][0]])*(danakb[i][konbinazix[zi][1]])*(danakb[i][konbinazix[zi][2]])*(1-danakb[i][sobx[zi][0]])*(1-danakb[i][sobx[zi][1]])
        laek3+= (danakl[i][konbinazix[zi][0]])*(danakl[i][konbinazix[zi][1]])*(danakl[i][konbinazix[zi][2]])*(1-danakl[i][sobx[zi][0]])*(1-danakl[i][sobx[zi][1]])
    for zo in range(0,10):
        benetakue2+= (danakb[i][konbinaz[zo][0]])*(danakb[i][konbinaz[zo][1]])*(1-danakb[i][sox[zo][0]])*(1-danakb[i][sox[zo][1]])*(1-danakb[i][sox[zo][2]])
        laek2+= (danakl[i][konbinaz[zo][0]])*(danakl[i][konbinaz[zo][1]])*(1-danakl[i][sox[zo][0]])*(1-danakl[i][sox[zo][1]])*(1-danakl[i][sox[zo][2]])
    for y in range(0,50):
        binomio=scipy.special.comb(estimazixue, y, exact=True)                
        laekue5+= ((1-laek5)**(estimazixue-y))*irabazixek5[y]*(laek5**y)*binomio
        laekue4+= ((1-laek4)**(estimazixue-y))*irabazixek4[y]*(laek4**y)*binomio
        laekue3+= ((1-laek3)**(estimazixue-y))*irabazixek3[y]*(laek3**y)*binomio
        laekue2+= ((1-laek2)**(estimazixue-y))*irabazixek2[y]*(laek2**y)*binomio
    maxi= (benetakue5 * laekue5) + (benetakue4 * laekue4) + (benetakue3 * laekue3) + (benetakue2 * laekue2)
    #if maxi>1:
    #    kantidadeOnak+=1
    for j in range (0,len(bukaera)):
        if maxi>bukaera[j][0]:
            bukaera.insert(j,[maxi, i, benetakue5])
            del bukaera[len(bukaera)-1]
            break

result(bukaera)
#print("Onak: ", kantidadeOnak)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))    