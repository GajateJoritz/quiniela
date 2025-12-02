import time
import numpy as np
from itertools import combinations
from datetime import timedelta
import xml.etree.ElementTree as ET
start_time = time.monotonic()
def faltadinak(bat,bi,hiru):
    for y in range(0,14):
        x=0
        for z in range(0,8):
            if bat[y]==bi[z]:
                break
            else:
                x=x+1
            if x==8:
                hiru.insert(0,bat[y])  

def result(lista):
    resultaue=["1","X","2"]
    plenue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
    for i in range(len(lista)):
        print(lista[i][0])
        print(resultaue[lista[i][1][0][0]],resultaue[lista[i][1][0][1]],resultaue[lista[i][1][0][2]],resultaue[lista[i][1][0][3]],resultaue[lista[i][1][0][4]],resultaue[lista[i][1][0][5]],resultaue[lista[i][1][0][6]],resultaue[lista[i][1][0][7]])
        print(lista[i][2])

bene=[]
lae=[]
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

estimazixue = 250000
#19.702.683 konbinazixo ta 3.003 irabazle, 6.561tik behin irabaztea

kantidadie=1
bukaera=[]
for i in range(10):
    bukaera.append([0,[]])
    
portzentaj=0.0
portzentaj2=0.0
emaitzek=[0,0,0,0,0,0,0,0,0]
emaitzek2=[0,0,0,0,0,0,0,0,0]
asmatzekue=0.0
asmatzekue2=0.0
bestiena=0.0
bestiena2=0.0

irabazixek=[0] * 180
konbodanak=[]


listie1=[bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]]
listie2=[lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]]
sobrak=[]
sobrak2=[]
irabazlik=[]
konbinazixuk=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]], 8)))
konbinazixuk2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]], 8)))

konbodanak=np.load("eligemotza.npy")

for x in range(3002,3003):
    for z in combinations(konbodanak,1):
        for xi in z:
            for a1 in range(0,3):
                faltadinak(listie1,konbinazixuk[x],sobrak)
                faltadinak(listie2,konbinazixuk2[x],sobrak2)
                lelenak=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],7)))
                bigarrenak=(list(combinations([sobrak2[0][a1],sobrak2[1][a1],sobrak2[2][a1],sobrak2[3][a1],sobrak2[4][a1],sobrak2[5][a1]],1)))
                danea=[]
                for aa in range(len(lelenak)):
                    for bb in range(len(bigarrenak)):
                        danea.append(lelenak[aa]+bigarrenak[bb])
                print(danea)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
