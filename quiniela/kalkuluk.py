import time
from datetime import timedelta
start_time = time.monotonic()
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
def faltad(bat,bi):
    for z in range(0,11):
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break
def falt(bat,bi):
    for z in range(0,10):
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break
pa1=26
pa2=57
pa3=14
pa4=3
pb1=5
pb2=7
pb3=48
pb4=20
pr1=27
pr2=49
pr3=16
pr4=8
ps1=9
ps2=26
ps3=43
ps4=22
lae1 = [90,6,4]
lae2 = [33,33,34]
lae3 = [52,29,19]
lae4 = [51,29,19]
lae5 = [23,29,48]
lae6 = [75,16,9]
lae7 = [45,33,22]
lae8 = [21,24,55]
lae9 = [72,20,8]
lae10 = [38,38,24]
lae11 = [57,31,12]
lae12 = [68,20,12]
lae13 = [62,26,12]
lae14 = [84,10,6]
bene1 = [76,16,8]
bene2 = [37,28,35]
bene3 = [44,29,27]
bene4 = [47,29,24]
bene5 = [28,33,39]
bene6 = [55,25,20]
bene7 = [50,26,24]
bene8 = [29,29,42]
bene9 = [52,30,18]
bene10 = [40,32,28]
bene11 = [52,28,20]
bene12 = [39,31,30]
bene13 = [53,27,20]
bene14 = [60,25,15]
bene15 = [pr1*ps1/10000,pr1*ps2/10000,pr1*ps3/10000,pr1*ps4/10000,pr2*ps1/10000,pr2*ps2/10000,pr2*ps3/10000,pr2*ps4/10000,pr3*ps1/10000,pr3*ps2/10000,pr3*ps3/10000,pr3*ps4/10000,pr4*ps1/10000,pr4*ps2/10000,pr4*ps3/10000,pr4*ps4/10000]
lae15 = [pa1*pb1,pa1*pb2,pa1*pb3,pa1*pb4,pa2*pb1,pa2*pb2,pa2*pb3,pa2*pb4,pa3*pb1,pa3*pb2,pa3*pb3,pa3*pb4,pa4*pb1,pa4*pb2,pa4*pb3,pa4*pb4]
#76.527.504 konbinazixo
estimazixue = 4800000
resultaue=["1","X","2"]
plenue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
portzentaj=0.0
portzentaj2=0.0
portzentaj3=0.0
from itertools import combinations

sobrak=[]
sobrak2=[]
sobrak3=[]
sobrak4=[]
sobrak5=[]
sobrak6=[]
sobrak7=[]
sobrak8=[]
emaitzek=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
konkretu=[0]
hamabostekue=0.0
hamalaukue=0.0
hamahirukue=0.0
hamabikue=0.0
hamaikakue=0.0
hamarrekue=0.0
hamabostekue2=0.0
hamalaukue2=0.0
hamahirukue2=0.0
hamabikue2=0.0
hamaikakue2=0.0
hamarrekue2=0.0
jeje=0.0
jeje2=0.0
jiji=0.0
jiji2=0.0
emaitzek2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
konkretu2=[0]
irabazixek10=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
irabazixek11=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
irabazixek12=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
irabazixek13=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
irabazixek14=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
irabazixek15=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for yi in range(0,10):
    if (estimazixue*0.075*0.75+2154000)/(yi+1)>40000:
        irabazixek15[yi]=((estimazixue*0.075*0.75+2154000)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek15[yi]=(estimazixue*0.075*0.75+2154000)/(yi+1)
    if (estimazixue*0.16*0.75)/(yi+1)>40000:
        irabazixek14[yi]=((estimazixue*0.16*0.75)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek14[yi]=(estimazixue*0.16*0.75)/(yi+1)
    
    if (estimazixue*0.075*0.75)/(yi+1)>40000:
        irabazixek13[yi]=((estimazixue*0.075*0.75)/(yi+1)-40000)*0.8+40000
        irabazixek12[yi]=((estimazixue*0.075*0.75)/(yi+1)-40000)*0.8+40000
        irabazixek11[yi]=((estimazixue*0.075*0.75)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek13[yi]=(estimazixue*0.075*0.75)/(yi+1)
        irabazixek12[yi]=(estimazixue*0.075*0.75)/(yi+1)
        irabazixek11[yi]=(estimazixue*0.075*0.75)/(yi+1)
    if (estimazixue*0.09*0.75)/(yi+1)>40000:
        irabazixek10[yi]=((estimazixue*0.09*0.75)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek10[yi]=(estimazixue*0.09*0.75)/(yi+1)
for a in range(0,1):
    for b in range(0,1):
        for c in range(0,1):
            for d in range(0,1):
                for e in range(0,1):
                    for f in range(0,1):
                        for g in range(0,1):
                            for h in range(0,1):
                                for i in range(0,1):
                                    for j in range(0,1):
                                        for k in range(0,1):
                                            for l in range(0,1):
                                                for m in range(0,1):
                                                    for n in range(0,3):
                                                        benetakue13ogexo=0.0
                                                        benetakue12ogexo=0.0
                                                        benetakue10=0.0
                                                        benetakue11=0.0
                                                        benetakue12=0.0
                                                        benetakue13=0.0
                                                        laek10=0.0
                                                        laek11=0.0
                                                        laek12=0.0
                                                        laek13=0.0
                                                        bene=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f]),(bene7[g]),(bene8[h]),(bene9[i]),(bene10[j]),(bene11[k]),(bene12[l]),(bene13[m]),(bene14[n])]
                                                        lae=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f]),(lae7[g]),(lae8[h]),(lae9[i]),(lae10[j]),(lae11[k]),(lae12[l]),(lae13[m]),(lae14[n])]
                                                        konbinazixuk=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]], 13)))
                                                        konbinazixuk2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]], 13)))
                                                        konbinazix=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]], 12)))
                                                        konbinazix2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]], 12)))
                                                        #konbinaz=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]], 11)))
                                                        #konbinaz2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]], 11)))
                                                        #konbin=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5],bene[6],bene[7],bene[8],bene[9],bene[10],bene[11],bene[12],bene[13]], 10)))
                                                        #konbin2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5],lae[6],lae[7],lae[8],lae[9],lae[10],lae[11],lae[12],lae[13]], 10)))
                                                        for z in range(0,14):
                                                            sobrak=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f]),(bene7[g]),(bene8[h]),(bene9[i]),(bene10[j]),(bene11[k]),(bene12[l]),(bene13[m]),(bene14[n])]
                                                            sobrak2=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f]),(lae7[g]),(lae8[h]),(lae9[i]),(lae10[j]),(lae11[k]),(lae12[l]),(lae13[m]),(lae14[n])]
                                                            faltadinak(sobrak,konbinazixuk[z])                                                       
                                                            faltadinak(sobrak2,konbinazixuk2[z])
                                                            benetakue13ogexo=benetakue13ogexo+(konbinazixuk[z][0]/100)*(konbinazixuk[z][1]/100)*(konbinazixuk[z][2]/100)*(konbinazixuk[z][3]/100)*(konbinazixuk[z][4]/100)*(konbinazixuk[z][5]/100)*(konbinazixuk[z][6]/100)*(konbinazixuk[z][7]/100)*(konbinazixuk[z][8]/100)*(konbinazixuk[z][9]/100)*(konbinazixuk[z][10]/100)*(konbinazixuk[z][11]/100)*(konbinazixuk[z][12]/100)
                                                            benetakue13=benetakue13+(konbinazixuk[z][0]/100)*(konbinazixuk[z][1]/100)*(konbinazixuk[z][2]/100)*(konbinazixuk[z][3]/100)*(konbinazixuk[z][4]/100)*(konbinazixuk[z][5]/100)*(konbinazixuk[z][6]/100)*(konbinazixuk[z][7]/100)*(konbinazixuk[z][8]/100)*(konbinazixuk[z][9]/100)*(konbinazixuk[z][10]/100)*(konbinazixuk[z][11]/100)*(konbinazixuk[z][12]/100)*(1-sobrak[0]/100)
                                                            laek13=laek13+(konbinazixuk2[z][0]/100)*(konbinazixuk2[z][1]/100)*(konbinazixuk2[z][2]/100)*(konbinazixuk2[z][3]/100)*(konbinazixuk2[z][4]/100)*(konbinazixuk2[z][5]/100)*(konbinazixuk2[z][6]/100)*(konbinazixuk2[z][7]/100)*(konbinazixuk2[z][8]/100)*(konbinazixuk2[z][9]/100)*(konbinazixuk2[z][10]/100)*(konbinazixuk2[z][11]/100)*(konbinazixuk2[z][12]/100)*(1-sobrak2[0]/100)                                                                     
                                                        for zi in range(0,91):
                                                            sobrak3=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f]),(bene7[g]),(bene8[h]),(bene9[i]),(bene10[j]),(bene11[k]),(bene12[l]),(bene13[m]),(bene14[n])]
                                                            sobrak4=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f]),(lae7[g]),(lae8[h]),(lae9[i]),(lae10[j]),(lae11[k]),(lae12[l]),(lae13[m]),(lae14[n])]
                                                            faltadin(sobrak3,konbinazix[zi])
                                                            faltadin(sobrak4,konbinazix2[zi])
                                                            benetakue12ogexo=benetakue12ogexo+(konbinazix[z][0]/100)*(konbinazix[z][1]/100)*(konbinazix[z][2]/100)*(konbinazix[z][3]/100)*(konbinazix[z][4]/100)*(konbinazix[z][5]/100)*(konbinazix[z][6]/100)*(konbinazix[z][7]/100)*(konbinazix[z][8]/100)*(konbinazix[z][9]/100)*(konbinazix[z][10]/100)*(konbinazix[z][11]/100)
                                                            benetakue12=benetakue12+(konbinazix[zi][0]/100)*(konbinazix[zi][1]/100)*(konbinazix[zi][2]/100)*(konbinazix[zi][3]/100)*(konbinazix[zi][4]/100)*(konbinazix[zi][5]/100)*(konbinazix[zi][6]/100)*(konbinazix[zi][7]/100)*(konbinazix[zi][8]/100)*(konbinazix[zi][9]/100)*(konbinazix[zi][10]/100)*(konbinazix[zi][11]/100)*(1-sobrak3[0]/100)*(1-sobrak3[1]/100)
                                                            laek12=laek12+(konbinazix2[zi][0]/100)*(konbinazix2[zi][1]/100)*(konbinazix2[zi][2]/100)*(konbinazix2[zi][3]/100)*(konbinazix2[zi][4]/100)*(konbinazix2[zi][5]/100)*(konbinazix2[zi][6]/100)*(konbinazix2[zi][7]/100)*(konbinazix2[zi][8]/100)*(konbinazix2[zi][9]/100)*(konbinazix2[zi][10]/100)*(konbinazix2[zi][11]/100)*(1-sobrak4[0]/100)*(1-sobrak4[1]/100)
                                                        #for zo in range(0,364):
                                                        #    sobrak5=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f]),(bene7[g]),(bene8[h]),(bene9[i]),(bene10[j]),(bene11[k]),(bene12[l]),(bene13[m]),(bene14[n])]
                                                        #    sobrak6=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f]),(lae7[g]),(lae8[h]),(lae9[i]),(lae10[j]),(lae11[k]),(lae12[l]),(lae13[m]),(lae14[n])]
                                                        #    faltad(sobrak5,konbinaz[zo])
                                                        #    faltad(sobrak6,konbinaz2[zo])
                                                        #    benetakue11=benetakue11+(konbinaz[zo][0]/100)*(konbinaz[zo][1]/100)*(konbinaz[zo][2]/100)*(konbinaz[zo][3]/100)*(konbinaz[zo][4]/100)*(konbinaz[zo][5]/100)*(konbinaz[zo][6]/100)*(konbinaz[zo][7]/100)*(konbinaz[zo][8]/100)*(konbinaz[zo][9]/100)*(konbinaz[zo][10]/100)*(1-sobrak5[0]/100)*(1-sobrak5[1]/100)*(1-sobrak5[2]/100)
                                                        #    laek11=laek11+(konbinaz2[zo][0]/100)*(konbinaz2[zo][1]/100)*(konbinaz2[zo][2]/100)*(konbinaz2[zo][3]/100)*(konbinaz2[zo][4]/100)*(konbinaz2[zo][5]/100)*(konbinaz2[zo][6]/100)*(konbinaz2[zo][7]/100)*(konbinaz2[zo][8]/100)*(konbinaz2[zo][9]/100)*(konbinaz2[zo][10]/100)*(1-sobrak6[0]/100)*(1-sobrak6[1]/100)*(1-sobrak6[2]/100)
                                                        #for zu in range(0,1001):
                                                        #    sobrak7=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f]),(bene7[g]),(bene8[h]),(bene9[i]),(bene10[j]),(bene11[k]),(bene12[l]),(bene13[m]),(bene14[n])]
                                                        #    sobrak8=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f]),(lae7[g]),(lae8[h]),(lae9[i]),(lae10[j]),(lae11[k]),(lae12[l]),(lae13[m]),(lae14[n])]
                                                        #    falt(sobrak7,konbin[zu])
                                                        #    falt(sobrak8,konbin2[zu])
                                                        #    benetakue10=benetakue10+(konbin[zu][0]/100)*(konbin[zu][1]/100)*(konbin[zu][2]/100)*(konbin[zu][3]/100)*(konbin[zu][4]/100)*(konbin[zu][5]/100)*(konbin[zu][6]/100)*(konbin[zu][7]/100)*(konbin[zu][8]/100)*(konbin[zu][9]/100)*(1-sobrak7[0]/100)*(1-sobrak7[1]/100)*(1-sobrak7[2]/100)*(1-sobrak7[3]/100)
                                                        #    laek10=laek10+(konbin2[zu][0]/100)*(konbin2[zu][1]/100)*(konbin2[zu][2]/100)*(konbin2[zu][3]/100)*(konbin2[zu][4]/100)*(konbin2[zu][5]/100)*(konbin2[zu][6]/100)*(konbin2[zu][7]/100)*(konbin2[zu][8]/100)*(konbin2[zu][9]/100)*(1-sobrak8[0]/100)*(1-sobrak8[1]/100)*(1-sobrak8[2]/100)*(1-sobrak8[3]/100)
                                                        
                                                        benetakue14=(bene1[a]/100)*(bene2[b]/100)*(bene3[c]/100)*(bene4[d]/100)*(bene5[e]/100)*(bene6[f]/100)*(bene7[g]/100)*(bene8[h]/100)*(bene9[i]/100)*(bene10[j]/100)*(bene11[k]/100)*(bene12[l]/100)*(bene13[m]/100)*(bene14[n]/100)
                                                        for o in range(0,2):
                                                            benetakue15=(bene1[a]/100)*(bene2[b]/100)*(bene3[c]/100)*(bene4[d]/100)*(bene5[e]/100)*(bene6[f]/100)*(bene7[g]/100)*(bene8[h]/100)*(bene9[i]/100)*(bene10[j]/100)*(bene11[k]/100)*(bene12[l]/100)*(bene13[m]/100)*(bene14[n]/100)*(bene15[o])   
                                                            laekue15=0.0
                                                            laekue14=0.0        
                                                            laekue13=0.0      
                                                            laekue12=0.0
                                                            laekue11=0.0
                                                            laekue10=0.0
                                                            for y in range(0,5):
                                                                #estimazixue+1 arte berez
                                                                #laekue10=laekue10+((1-(laek10))**(estimazixue-y))*irabazixek10[y]*(laek10**y)
                                                                #laekue11=laekue11+((1-(laek11))**(estimazixue-y))*irabazixek11[y]*(laek11**y)
                                                                laekue12=laekue12+((1-(laek12))**(estimazixue-y))*irabazixek12[y]*(laek12**y)
                                                                laekue13=laekue13+((1-(laek13))**(estimazixue-y))*irabazixek13[y]*(laek13**y)
                                                                laekue14=laekue14+((1-(lae1[a]/100)*(lae2[b]/100)*(lae3[c]/100)*(lae4[d]/100)*(lae5[e]/100)*(lae6[f]/100)*(lae7[g]/100)*(lae8[h]/100)*(lae9[i]/100)*(lae10[j]/100)*(lae11[k]/100)*(lae12[l]/100)*(lae13[m]/100)*(lae14[n]/100))**(estimazixue-y))*irabazixek14[y]*(((lae1[a]/100)*(lae2[b]/100)*(lae3[c]/100)*(lae4[d]/100)*(lae5[e]/100)*(lae6[f]/100)*(lae7[g]/100)*(lae8[h]/100)*(lae9[i]/100)*(lae10[j]/100)*(lae11[k]/100)*(lae12[l]/100)*(lae13[m]/100)*(lae14[n]/100))**y)
                                                                laekue15=laekue15+((1-(lae15[o]/10000)*(lae1[a]/100)*(lae2[b]/100)*(lae3[c]/100)*(lae4[d]/100)*(lae5[e]/100)*(lae6[f]/100)*(lae7[g]/100)*(lae8[h]/100)*(lae9[i]/100)*(lae10[j]/100)*(lae11[k]/100)*(lae12[l]/100)*(lae13[m]/100)*(lae14[n]/100))**(estimazixue-y))*irabazixek15[y]*(((lae15[o]/10000)*(lae1[a]/100)*(lae2[b]/100)*(lae3[c]/100)*(lae4[d]/100)*(lae5[e]/100)*(lae6[f]/100)*(lae7[g]/100)*(lae8[h]/100)*(lae9[i]/100)*(lae10[j]/100)*(lae11[k]/100)*(lae12[l]/100)*(lae13[m]/100)*(lae14[n]/100))**y)
                                                            maxi = ((benetakue15*laekue15)+(benetakue14*laekue14)+(benetakue13*laekue13)+(benetakue12*laekue12)+(benetakue11*laekue11)+(benetakue10*laekue10))/0.75
                                                            if benetakue12ogexo>0.001:
                                                                if maxi>portzentaj:
                                                                    portzentaj3=portzentaj2
                                                                    portzentaj2=portzentaj
                                                                    portzentaj=maxi
                                                                    hamabostekue3=hamabostekue2
                                                                    hamabostekue2=hamabostekue
                                                                    hamabostekue=(benetakue15*laekue15)/0.75
                                                                    hamalaukue3=hamalaukue2
                                                                    hamalaukue2=hamalaukue
                                                                    hamalaukue=(benetakue14*laekue14)/0.75
                                                                    hamahirukue3=hamahirukue2
                                                                    hamahirukue2=hamahirukue
                                                                    hamahirukue=(benetakue13*laekue13)/0.75
                                                                    hamabikue3=hamabikue2
                                                                    hamabikue2=hamabikue
                                                                    hamabikue=(benetakue12*laekue12)/0.75
                                                                    hamaikakue3=hamaikakue2
                                                                    hamaikakue2=hamaikakue
                                                                    hamaikakue=(benetakue11*laekue11)/0.75
                                                                    hamarrekue3=hamarrekue2
                                                                    hamarrekue2=hamarrekue
                                                                    hamarrekue=(benetakue10*laekue10)/0.75
                                                                    emaitzek3=emaitzek2
                                                                    emaitzek2=emaitzek
                                                                    emaitzek=[a,b,c,d,e,f,g,h,i,j,k,l,m,n]
                                                                    konkretu3=konkretu2
                                                                    konkretu2=konkretu
                                                                    konkretu=[o]
                                                                    jeje3=jeje2
                                                                    jeje2=jeje
                                                                    jeje=benetakue13ogexo
                                                                    jiji3=jiji2
                                                                    jiji2=jiji
                                                                    jiji=benetakue12ogexo
                                                                elif maxi>portzentaj2:
                                                                    portzentaj3=portzentaj2
                                                                    portzentaj2=maxi
                                                                    hamabostekue3=hamabostekue2
                                                                    hamabaostekue2=(benetakue15*laekue15)/0.75
                                                                    hamalaukue3=hamalaukue2
                                                                    hamalaukue2=(benetakue14*laekue14)/0.75
                                                                    hamahirukue3=hamahirukue2
                                                                    hamahirukue2=(benetakue13*laekue13)/0.75
                                                                    hamabikue3=hamabikue2
                                                                    hamabikue2=(benetakue12*laekue12)/0.75
                                                                    hamaikakue3=hamaikakue2
                                                                    hamaikakaue2=(benetakue11*laekue11)/0.75
                                                                    hamarrekue3=hamarrekue2
                                                                    hamarrekue2=(benetakue10*laekue10)/0.75
                                                                    emaitzek3=emaitzek2
                                                                    emaitzek2=[a,b,c,d,e,f,g,h,i,j,k,l,m,n]
                                                                    konkretu3=konkretu2
                                                                    konkretu2=[o]
                                                                    jeje3=jeje2
                                                                    jeje2=benetakue13ogexo
                                                                    jiji3=jiji2
                                                                    jiji2=benetakue12ogexo
                                                                elif maxi>portzentaj3:
                                                                    portzentaj3=maxi
                                                                    hamabaostekue3=(benetakue15*laekue15)/0.75
                                                                    hamalaukue3=(benetakue14*laekue14)/0.75
                                                                    hamahirukue3=(benetakue13*laekue13)/0.75
                                                                    hamabikue3=(benetakue12*laekue12)/0.75
                                                                    hamaikakaue3=(benetakue11*laekue11)/0.75
                                                                    hamarrekue3=(benetakue10*laekue10)/0.75
                                                                    emaitzek3=[a,b,c,d,e,f,g,h,i,j,k,l,m,n]
                                                                    konkretu3=[o]
                                                                    jeje3=benetakue13ogexo
                                                                    jiji3=benetakue12ogexo
print(portzentaj)
print(resultaue[emaitzek[0]],resultaue[emaitzek[1]],resultaue[emaitzek[2]],resultaue[emaitzek[3]],resultaue[emaitzek[4]],resultaue[emaitzek[5]],resultaue[emaitzek[6]],resultaue[emaitzek[7]],resultaue[emaitzek[8]],resultaue[emaitzek[9]],resultaue[emaitzek[10]],resultaue[emaitzek[11]],resultaue[emaitzek[12]],resultaue[emaitzek[13]])
print(plenue[konkretu[0]])
print("15ekuena:")
print(hamabostekue)
print("14kuena:")
print(hamalaukue)
print("13kuena:")
print(hamahirukue)
print("12kuena:")
print(hamabikue)
print("11kuena:")
print(hamaikakue)
print("10kuena:")
print(hamarrekue)
print("13ogexo:")
print(jeje)
print("12ogexo:")
print(jiji)
print()
print(portzentaj2)
print(resultaue[emaitzek2[0]],resultaue[emaitzek2[1]],resultaue[emaitzek2[2]],resultaue[emaitzek2[3]],resultaue[emaitzek2[4]],resultaue[emaitzek2[5]],resultaue[emaitzek2[6]],resultaue[emaitzek2[7]],resultaue[emaitzek2[8]],resultaue[emaitzek2[9]],resultaue[emaitzek2[10]],resultaue[emaitzek2[11]],resultaue[emaitzek2[12]],resultaue[emaitzek2[13]])
print(plenue[konkretu2[0]])
print("15ekuena:")
print(hamabostekue2)
print("14kuena:")
print(hamalaukue2)
print("13kuena:")
print(hamahirukue2)
print("12kuena:")
print(hamabikue2)
print("11kuena:")
print(hamaikakue2)
print("10kuena:")
print(hamarrekue2)
print("13ogexo:")
print(jeje2)
print("12ogexo:")
print(jiji2)
print()
print()
print(portzentaj3)
print(resultaue[emaitzek3[0]],resultaue[emaitzek3[1]],resultaue[emaitzek3[2]],resultaue[emaitzek3[3]],resultaue[emaitzek3[4]],resultaue[emaitzek3[5]],resultaue[emaitzek3[6]],resultaue[emaitzek3[7]],resultaue[emaitzek3[8]],resultaue[emaitzek3[9]],resultaue[emaitzek3[10]],resultaue[emaitzek3[11]],resultaue[emaitzek3[12]],resultaue[emaitzek3[13]])
print(plenue[konkretu3[0]])
print("15ekuena:")
print(hamabostekue3)
print("14kuena:")
print(hamalaukue3)
print("13kuena:")
print(hamahirukue3)
print("12kuena:")
print(hamabikue3)
print("11kuena:")
print(hamaikakue3)
print("10kuena:")
print(hamarrekue3)
print("13ogexo:")
print(jeje3)
print("12ogexo:")
print(jiji3)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))