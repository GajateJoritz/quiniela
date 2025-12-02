import time
from datetime import timedelta
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
print("ipa")
estimazixue = 200000
#19.702.683 konbinazixo ta 3.003 irabazle, 6.561tik behin irabaztea
lae1 = [39,32,29]
lae2 = [62,23,15]
lae3 = [22,20,58]
lae4 = [46,35,19]
lae5 = [50,32,18]
lae6 = [28,36,36]
lae7 = [55,26,19]
lae8 = [86,8,6]
lae9 = [45,29,26]
lae10 = [26,31,43]
lae11 = [44,29,27]
lae12 = [82,11,7]
lae13 = [65,24,11]
lae14 = [29,37,34]
bene1 = [33.08,29.62,37.3]
bene2 = [43.55,30.36,26.09]
bene3 = [27.07,24.39,48.54]
bene4 = [42.72,29.58,27.7]
bene5 = [48.8,25.7,25.5]
bene6 = [35.49,31.12,33.39]
bene7 = [41.41,29.03,29.56]
bene8 = [70.36,18.61,11.03]
bene9 = [45.12,30.13,24.75]
bene10 = [40.47,29.01,30.52]
bene11 = [34.85,31.9,33.25]
bene12 = [53.34,26.74,19.92]
bene13 = [54.24,26.66,19.1]
bene14 = [41.63,31.29,27.08]
resultaue=["1","X","2"]
portzentaj=0.0
portzentaj2=0.0
emaitzek=[0,0,0,0,0,0,0,0,0]
emaitzek2=[0,0,0,0,0,0,0,0,0]
asmatzekue=0.0
asmatzekue2=0.0
bestiena=0.0
bestiena2=0.0
irabazixek=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
from itertools import combinations

listie1=[bene1,bene2,bene3,bene4,bene5,bene6,bene7,bene8,bene9,bene10,bene11,bene12,bene13,bene14]
listie2=[lae1,lae2,lae3,lae4,lae5,lae6,lae7,lae8,lae9,lae10,lae11,lae12,lae13,lae14]
sobrak=[]
sobrak2=[]
irabazlik=[]
konbinazixuk=(list(combinations([bene1,bene2,bene3,bene4,bene5,bene6,bene7,bene8,bene9,bene10,bene11,bene12,bene13,bene14], 8)))
konbinazixuk2=(list(combinations([lae1,lae2,lae3,lae4,lae5,lae6,lae7,lae8,lae9,lae10,lae11,lae12,lae13,lae14], 8)))

for yi in range(0,10):
    if (estimazixue*0.5*0.55)/(yi+1)>40000:
        irabazixek[yi]=((estimazixue*0.5*0.55)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek[yi]=(estimazixue*0.5*0.55)/(yi+1)
for x in range(1339,1340):
    #berez 3003 arte
    faltadinak(listie1,konbinazixuk[x],sobrak)
    faltadinak(listie2,konbinazixuk2[x],sobrak2)
    for a in range(0,1):
        #berez hauek danak 3 arte
        for b in range(0,1):
            for c in range(0,1):
                for d in range(0,3):
                    for e in range(0,1):
                        for f in range(0,1):
                            for g in range(0,1):
                                for h in range(0,1):
                                    laetotala=0.0
                                    laekue=0.0
                                    benetakue=(konbinazixuk[x][0][a]/100)*(konbinazixuk[x][1][b]/100)*(konbinazixuk[x][2][c]/100)*(konbinazixuk[x][3][d]/100)*(konbinazixuk[x][4][e]/100)*(konbinazixuk[x][5][f]/100)*(konbinazixuk[x][6][g]/100)*(konbinazixuk[x][7][h]/100)
                                    for a1 in range(0,3):
                                        for a2 in range(0,3):
                                            for a3 in range(0,3):
                                                for a4 in range(0,3):
                                                    for a5 in range(0,3):
                                                        for a6 in range(0,3):
                                                            irabazlik=(list(combinations([konbinazixuk2[x][0][a],konbinazixuk2[x][1][b],konbinazixuk2[x][2][c],konbinazixuk2[x][3][d],konbinazixuk2[x][4][e],konbinazixuk2[x][5][f],konbinazixuk2[x][6][g],konbinazixuk2[x][7][h],sobrak2[0][a1],sobrak2[1][a2],sobrak2[2][a3],sobrak2[3][a4],sobrak2[4][a5],sobrak2[5][a6]],8)))
                                                            laek=0.0
                                                            for a7 in range(0,3003):
                                                                laek+=(irabazlik[a7][0]/100)*(irabazlik[a7][1]/100)*(irabazlik[a7][2]/100)*(irabazlik[a7][3]/100)*(irabazlik[a7][4]/100)*(irabazlik[a7][5]/100)*(irabazlik[a7][6]/100)*(irabazlik[a7][7]/100)/3003#Ya behin gure 8 partiduk asmaute ta beste sei partido hoxek jokaute edozeiñek eligeochue asmatzeko portzentajie, 1/3003 portzentajie dauke konbo bakoitze aukeratzeko ta horko bakoitzin 1/3^8 portzentajie mas o menos
                                                            laekue+=(sobrak[0][a1]/100)*(sobrak[1][a2]/100)*(sobrak[2][a3]/100)*(sobrak[3][a4]/100)*(sobrak[4][a5]/100)*(sobrak[5][a6]/100)*laek
                                    for y in range(0,10):
                                        laetotala=laetotala+((1-laekue)**(estimazixue-y))*irabazixek[y]*(laekue**y)
                                    portzentajie =benetakue*laetotala/0.5
                                    if benetakue>0.000:
                                        if portzentajie>portzentaj:
                                            portzentaj2=portzentaj
                                            portzentaj=portzentajie
                                            asmatzekue2=asmatzekue
                                            asmatzekue=benetakue
                                            emaitzek2=emaitzek
                                            emaitzek=[a,b,c,d,e,f,g,h,konbinazixuk[x]]
                                            bestiena2=bestiena
                                            bestiena=laetotala
                                        elif portzentajie>portzentaj2:
                                            portzentaj2=portzentajie
                                            asmatzekue2=benetakue
                                            bestiena2=laetotala
                                            emaitzek2=[a,b,c,d,e,f,g,h,konbinazixuk[x]]
print(portzentaj)
print(resultaue[emaitzek[0]],resultaue[emaitzek[1]],resultaue[emaitzek[2]],resultaue[emaitzek[3]],resultaue[emaitzek[4]],resultaue[emaitzek[5]],resultaue[emaitzek[6]],resultaue[emaitzek[7]],emaitzek[8])
print("Asmatzeko probabilidadie:")
print(asmatzekue)
print("Bestie")
print(bestiena)
print()
print(portzentaj2)
print(resultaue[emaitzek2[0]],resultaue[emaitzek2[1]],resultaue[emaitzek2[2]],resultaue[emaitzek2[3]],resultaue[emaitzek2[4]],resultaue[emaitzek2[5]],resultaue[emaitzek2[6]],resultaue[emaitzek2[7]],emaitzek2[8])
print("Asmatzeko probabilidadie:")
print(asmatzekue2)
print("Bestie:")
print(bestiena2)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))    

