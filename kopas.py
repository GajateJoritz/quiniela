import time
import numpy as np
import scipy.special
from itertools import combinations
from datetime import timedelta
import xml.etree.ElementTree as ET
from konbinazixueErakutsi import konbinazixuerakutsi
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
        print(resultaue[lista[i][1][0]],resultaue[lista[i][1][1]],resultaue[lista[i][1][2]],resultaue[lista[i][1][3]],resultaue[lista[i][1][4]],resultaue[lista[i][1][5]],resultaue[lista[i][1][6]],resultaue[lista[i][1][7]])
        konbinazixuerakutsi(lista[i][2])
        print(lista[i][3])
        print()
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
"""
with open("estimacion.txt", mode="r", encoding="utf-8-sig") as datos:
    valores=[]
    for linea in datos:
        linea=linea.replace(",",".")
        valores.append([float(x)/100 for x in linea.split()])
for i in range(0,14):
    lae.append([valores[i][0],valores[i][1],valores[i][2]])


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
for i in range(90):
    bukaera.append([0,[]])
    
portzentaj=0.0
portzentaj2=0.0
emaitzek=[0,0,0,0,0,0,0,0,0]
emaitzek2=[0,0,0,0,0,0,0,0,0]
asmatzekue=0.0
asmatzekue2=0.0
bestiena=0.0
bestiena2=0.0

irabazixek=[0] * (1000)
konbodanak=[]


listie1=[bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]]
listie2=[lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]]
sobrak=[]
sobrak2=[]
irabazlik=[]
konbinazixuk=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]], 8)))
konbinazixuk2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]], 8)))

for yi in range(0,1000):
    if (estimazixue*0.5*0.55)/(yi+1)>40000:
        irabazixek[yi]=((estimazixue*0.5*0.55)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek[yi]=(estimazixue*0.5*0.55)/(yi+1)

konbodanak=np.load("kopis.npy", allow_pickle=True)  
print(len(konbodanak))                     
for k in range(len(konbodanak)):
    x=konbodanak[k][0][8]
    #berez 3003 arte
    faltadinak(listie1,konbinazixuk[x],sobrak)
    faltadinak(listie2,konbinazixuk2[x],sobrak2)
    for xi in konbodanak[k]:
        laetotala=0.0
        laekue=0.0
        laek=0.0
        benetakue=(konbinazixuk[x][0][xi[0]])*(konbinazixuk[x][1][xi[1]])*(konbinazixuk[x][2][xi[2]])*(konbinazixuk[x][3][xi[3]])*(konbinazixuk[x][4][xi[4]])*(konbinazixuk[x][5][xi[5]])*(konbinazixuk[x][6][xi[6]])*(konbinazixuk[x][7][xi[7]])
        laekue+=(konbinazixuk2[x][0][xi[0]])*(konbinazixuk2[x][1][xi[1]])*(konbinazixuk2[x][2][xi[2]])*(konbinazixuk2[x][3][xi[3]])*(konbinazixuk2[x][4][xi[4]])*(konbinazixuk2[x][5][xi[5]])*(konbinazixuk2[x][6][xi[6]])*(konbinazixuk2[x][7][xi[7]])/3003
        lelenak7=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],7)))
        lelenak6=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],6)))
        lelenak5=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],5)))
        lelenak4=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],4)))
        lelenak3=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],3)))
        lelenak2=(list(combinations([konbinazixuk2[x][0][xi[0]],konbinazixuk2[x][1][xi[1]],konbinazixuk2[x][2][xi[2]],konbinazixuk2[x][3][xi[3]],konbinazixuk2[x][4][xi[4]],konbinazixuk2[x][5][xi[5]],konbinazixuk2[x][6][xi[6]],konbinazixuk2[x][7][xi[7]]],2)))
        bigarrenak6=(list(combinations([sobrak2[0],sobrak2[1],sobrak2[2],sobrak2[3],sobrak2[4],sobrak2[5]],6)))
        bigarrenak5=(list(combinations([sobrak2[0],sobrak2[1],sobrak2[2],sobrak2[3],sobrak2[4],sobrak2[5]],5)))
        bigarrenak4=(list(combinations([sobrak2[0],sobrak2[1],sobrak2[2],sobrak2[3],sobrak2[4],sobrak2[5]],4)))
        bigarrenak3=(list(combinations([sobrak2[0],sobrak2[1],sobrak2[2],sobrak2[3],sobrak2[4],sobrak2[5]],3)))
        bigarrenak2=(list(combinations([sobrak2[0],sobrak2[1],sobrak2[2],sobrak2[3],sobrak2[4],sobrak2[5]],2)))
        bigarrenak1=(list(combinations([sobrak2[0],sobrak2[1],sobrak2[2],sobrak2[3],sobrak2[4],sobrak2[5]],1)))
        bigar6=(list(combinations([sobrak[0],sobrak[1],sobrak[2],sobrak[3],sobrak[4],sobrak[5]],6)))
        bigar5=(list(combinations([sobrak[0],sobrak[1],sobrak[2],sobrak[3],sobrak[4],sobrak[5]],5)))
        bigar4=(list(combinations([sobrak[0],sobrak[1],sobrak[2],sobrak[3],sobrak[4],sobrak[5]],4)))
        bigar3=(list(combinations([sobrak[0],sobrak[1],sobrak[2],sobrak[3],sobrak[4],sobrak[5]],3)))
        bigar2=(list(combinations([sobrak[0],sobrak[1],sobrak[2],sobrak[3],sobrak[4],sobrak[5]],2)))
        bigar1=(list(combinations([sobrak[0],sobrak[1],sobrak[2],sobrak[3],sobrak[4],sobrak[5]],1)))
        for a1 in range(0,3):                           
            danea=[]
            for aa in range(len(lelenak7)):
                for bb in range(len(bigarrenak1)):
                    danea.append(lelenak7[aa]+(bigarrenak1[bb][0][a1],))
                    laek+=(danea[aa*6+bb][0])*(danea[aa*6+bb][1])*(danea[aa*6+bb][2])*(danea[aa*6+bb][3])*(danea[aa*6+bb][4])*(danea[aa*6+bb][5])*(danea[aa*6+bb][6])*(danea[aa*6+bb][7])*(bigar1[bb][0][a1])/3003
            laekue+= laek
            for a2 in range(0,3):
                danea=[] 
                laek=0.0      
                for aa in range(len(lelenak6)):
                    for bb in range(len(bigarrenak2)):
                        danea.append(lelenak6[aa]+(bigarrenak2[bb][0][a1],)+(bigarrenak2[bb][1][a2],))
                        laek+=(danea[aa*15+bb][0])*(danea[aa*15+bb][1])*(danea[aa*15+bb][2])*(danea[aa*15+bb][3])*(danea[aa*15+bb][4])*(danea[aa*15+bb][5])*(danea[aa*15+bb][6])*(danea[aa*15+bb][7])*(bigar2[bb][0][a1])*(bigar2[bb][1][a2])/3003
                laekue+= laek
                for a3 in range(0,3):
                    danea=[]  
                    laek=0.0      
                    for aa in range(len(lelenak5)):
                        for bb in range(len(bigarrenak3)):
                            danea.append(lelenak5[aa]+(bigarrenak3[bb][0][a1],)+(bigarrenak3[bb][1][a2],)+(bigarrenak3[bb][2][a3],))
                            laek+=(danea[aa*20+bb][0])*(danea[aa*20+bb][1])*(danea[aa*20+bb][2])*(danea[aa*20+bb][3])*(danea[aa*20+bb][4])*(danea[aa*20+bb][5])*(danea[aa*20+bb][6])*(danea[aa*20+bb][7])*(bigar3[bb][0][a1])*(bigar3[bb][1][a2])*(bigar3[bb][2][a3])/3003
                    laekue+= laek
                    for a4 in range(0,3):
                        danea=[]   
                        laek=0.0     
                        for aa in range(len(lelenak4)):
                            for bb in range(len(bigarrenak4)):
                                danea.append(lelenak4[aa]+(bigarrenak4[bb][0][a1],)+(bigarrenak4[bb][1][a2],)+(bigarrenak4[bb][2][a3],)+(bigarrenak4[bb][3][a4],))
                                laek+=(danea[aa*15+bb][0])*(danea[aa*15+bb][1])*(danea[aa*15+bb][2])*(danea[aa*15+bb][3])*(danea[aa*15+bb][4])*(danea[aa*15+bb][5])*(danea[aa*15+bb][6])*(danea[aa*15+bb][7])*(bigar4[bb][0][a1])*(bigar4[bb][1][a2])*(bigar4[bb][2][a3])*(bigar4[bb][3][a4])/3003
                        laekue+= laek
                        for a5 in range(0,3):
                            danea=[]  
                            laek=0.0      
                            for aa in range(len(lelenak3)):
                                for bb in range(len(bigarrenak5)):
                                    danea.append(lelenak3[aa]+(bigarrenak5[bb][0][a1],)+(bigarrenak5[bb][1][a2],)+(bigarrenak5[bb][2][a3],)+(bigarrenak5[bb][3][a4],)+(bigarrenak5[bb][4][a5],))
                                    laek+=(danea[aa*6+bb][0])*(danea[aa*6+bb][1])*(danea[aa*6+bb][2])*(danea[aa*6+bb][3])*(danea[aa*6+bb][4])*(danea[aa*6+bb][5])*(danea[aa*6+bb][6])*(danea[aa*6+bb][7])*(bigar5[bb][0][a1])*(bigar5[bb][1][a2])*(bigar5[bb][2][a3])*(bigar5[bb][3][a4])*(bigar5[bb][4][a5])/3003
                            laekue+= laek
                            for a6 in range(0,3):
                                danea=[]  
                                laek=0.0      
                                for aa in range(len(lelenak2)):
                                    for bb in range(len(bigarrenak6)):
                                        danea.append(lelenak2[aa]+(bigarrenak6[bb][0][a1],)+(bigarrenak6[bb][1][a2],)+(bigarrenak6[bb][2][a3],)+(bigarrenak6[bb][3][a4],)+(bigarrenak6[bb][4][a5],)+(bigarrenak6[bb][5][a6],))
                                        laek+=(danea[aa][0])*(danea[aa][1])*(danea[aa][2])*(danea[aa][3])*(danea[aa][4])*(danea[aa][5])*(danea[aa][6])*(danea[aa][7])*(bigar6[bb][0][a1])*(bigar6[bb][1][a2])*(bigar6[bb][2][a3])*(bigar6[bb][3][a4])*(bigar6[bb][4][a5])*(bigar6[bb][5][a6])/3003
                                laekue+= laek
        for y in range(0,50):
            binomio=scipy.special.comb(estimazixue, y, exact=True)
            laetotala+= ((1-laekue)**(estimazixue-y))*irabazixek[y]*(laekue**y)*binomio 
        maxi =(benetakue*laetotala)/0.5
        #if maxi>bukaera[len(bukaera)-1][0]:
        if maxi>bukaera[k][0]:
            bukaera.insert(k,[maxi,xi,x, benetakue])
            del bukaera[k+1]
def sorteau(a):
    return a[0]
bukaera.sort(key=sorteau,reverse=True)
result(bukaera)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))