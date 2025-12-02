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
        print(lista[i][0])
        print(resultaue[lista[i][1][0]],resultaue[lista[i][1][1]],resultaue[lista[i][1][2]],resultaue[lista[i][1][3]],resultaue[lista[i][1][4]],resultaue[lista[i][1][5]])
        print(lista[i][2])
        f.write(resul[lista[i][1][0]] + resul[lista[i][1][1]] + resul[lista[i][1][2]] + resul[lista[i][1][3]] + resul[lista[i][1][4]] + resul[lista[i][1][5]] + "\n")
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
ko = list(combinations(numeros, 5))
kon = list(combinations(numeros, 4))
konb = list(combinations(numeros, 3))
konbi = list(combinations(numeros, 2))
num = set(numeros)
sobrx=[]
sobx=[]
sox=[]
for elem in kon:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobrx.append(faltantes) 
for elem in konb:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sobx.append(faltantes)
for elem in konbi:
    conjunto = set(elem)
    faltantes= list(num - conjunto)
    sox.append(faltantes)
konbin = list(combinations(bene, 5))
konbin2 = list(combinations(lae, 5))
konbinazixuk = list(combinations(bene, 4))
konbinazixuk2 = list(combinations(lae, 4))
konbinazix = list(combinations(bene, 3))
konbinazix2 = list(combinations(lae, 3))    
konbinaz = list(combinations(bene, 2))
konbinaz2 = list(combinations(lae, 2))      
for xi in konbodanak:
    #if diferent5(xi):
    #    continue
    benetakue6=(bene[0][xi[0]])*(bene[1][xi[1]])*(bene[2][xi[2]])*(bene[3][xi[3]])*(bene[4][xi[4]])*(bene[5][xi[5]])
    #if benetakue6 < 0.0000001:
    #    continue
    laek6=(lae[0][xi[0]])*(lae[1][xi[1]])*(lae[2][xi[2]])*(lae[3][xi[3]])*(lae[4][xi[4]])*(lae[5][xi[5]])
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
        benetakue5+= (konbin[zu][0][xi[ko[zu][0]]])*(konbin[zu][1][xi[ko[zu][1]]])*(konbin[zu][2][xi[ko[zu][2]]])*(konbin[zu][3][xi[ko[zu][3]]])*(konbin[zu][4][xi[ko[zu][4]]])*(1-bene[sobrkx[zu]][xi[sobrkx[zu]]])
        laek5+= (konbin2[zu][0][xi[ko[zu][0]]])*(konbin2[zu][1][xi[ko[zu][1]]])*(konbin2[zu][2][xi[ko[zu][2]]])*(konbin2[zu][3][xi[ko[zu][3]]])*(konbin2[zu][4][xi[ko[zu][4]]])*(1-lae[sobrkx[zu]][xi[sobrkx[zu]]])
    for z in range(0,15):
        benetakue4+= (konbinazixuk[z][0][xi[kon[z][0]]])*(konbinazixuk[z][1][xi[kon[z][1]]])*(konbinazixuk[z][2][xi[kon[z][2]]])*(konbinazixuk[z][3][xi[kon[z][3]]])*(1-bene[sobrx[z][0]][xi[sobrx[z][0]]])*(1-bene[sobrx[z][1]][xi[sobrx[z][1]]])
        laek4+= (konbinazixuk2[z][0][xi[kon[z][0]]])*(konbinazixuk2[z][1][xi[kon[z][1]]])*(konbinazixuk2[z][2][xi[kon[z][2]]])*(konbinazixuk2[z][3][xi[kon[z][3]]])*(1-lae[sobrx[z][0]][xi[sobrx[z][0]]])*(1-lae[sobrx[z][1]][xi[sobrx[z][1]]])
    for zi in range(0,20):
        benetakue3+= (konbinazix[zi][0][xi[konb[zi][0]]])*(konbinazix[zi][1][xi[konb[zi][1]]])*(konbinazix[zi][2][xi[konb[zi][2]]])*(1-bene[sobx[zi][0]][xi[sobx[zi][0]]])*(1-bene[sobx[zi][1]][xi[sobx[zi][1]]])*(1-bene[sobx[zi][2]][xi[sobx[zi][2]]])
        laek3+= (konbinazix2[zi][0][xi[konb[zi][0]]])*(konbinazix2[zi][1][xi[konb[zi][1]]])*(konbinazix2[zi][2][xi[konb[zi][2]]])*(1-lae[sobx[zi][0]][xi[sobx[zi][0]]])*(1-lae[sobx[zi][1]][xi[sobx[zi][1]]])*(1-lae[sobx[zi][2]][xi[sobx[zi][2]]])
    for zo in range(0,15):
        benetakue2+= (konbinaz[zo][0][xi[konbi[zo][0]]])*(konbinaz[zo][1][xi[konbi[zo][1]]])*(1-bene[sox[zo][0]][xi[sox[zo][0]]])*(1-bene[sox[zo][1]][xi[sox[zo][1]]])*(1-bene[sox[zo][2]][xi[sox[zo][2]]])*(1-bene[sox[zo][3]][xi[sox[zo][3]]])
        laek2+= (konbinaz2[zo][0][xi[konbi[zo][0]]])*(konbinaz2[zo][1][xi[konbi[zo][1]]])*(1-lae[sox[zo][0]][xi[sox[zo][0]]])*(1-lae[sox[zo][1]][xi[sox[zo][1]]])*(1-lae[sox[zo][2]][xi[sox[zo][2]]])*(1-lae[sox[zo][3]][xi[sox[zo][3]]])
    for y in range(0,50):
        binomio=scipy.special.comb(estimazixue, y, exact=True)
        laekue6+= ((1-laek6)**(estimazixue-y))*irabazixek6[y]*(laek6**y)*binomio                   
        laekue5+= ((1-laek5)**(estimazixue-y))*irabazixek5[y]*(laek5**y)*binomio
        laekue4+= ((1-laek4)**(estimazixue-y))*irabazixek4[y]*(laek4**y)*binomio
        laekue3+= ((1-laek3)**(estimazixue-y))*irabazixek3[y]*(laek3**y)*binomio
        laekue2+= ((1-laek2)**(estimazixue-y))*irabazixek2[y]*(laek2**y)*binomio
    maxi= (benetakue6 * laekue6) + (benetakue5 * laekue5) + (benetakue4 * laekue4) + (benetakue3 * laekue3) + (benetakue2 * laekue2)
    #if maxi>1:
    #    kantidadeOnak+=1
    for i in range (0,len(bukaera)):
        if maxi>bukaera[i][0]:
            bukaera.insert(i,[maxi, xi, benetakue6])
            del bukaera[len(bukaera)-1]
            break

result(bukaera)
#print("Onak: ", kantidadeOnak)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))    