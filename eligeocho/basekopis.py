import time
import pickle
from datetime import timedelta
from itertools import combinations
start_time = time.monotonic()
import numpy as np
class Baseelige():
    def __init__(self):
        self.konbos=[]
    def konbuk(self):
        jiji=(list(combinations([1,2,3,4,5,6,7,8,9,10,11,12,13,14], 8)))
        kont=0
        zein= [601, 602, 599, 596, 467]
        #zein=list(range(3000,3002))
        for k in (range(5)):
            i=zein[k]
            self.konbos.append([])
            auk=[]
            for j in range(len(jiji[i])):
                if jiji[i][j] == 1:
                    auk.append([0])
                if jiji[i][j] == 2:
                    auk.append([0])
                if jiji[i][j] == 3:
                    auk.append([0])
                if jiji[i][j] == 4:
                    auk.append([0])
                if jiji[i][j] == 5:
                    auk.append([0])
                if jiji[i][j] == 6:
                    auk.append([0])
                if jiji[i][j] == 7:
                    auk.append([0])
                if jiji[i][j] == 8:
                    auk.append([0])
                if jiji[i][j] == 9:
                    auk.append([2])
                if jiji[i][j] == 10:
                    auk.append([0])
                if jiji[i][j] == 11:
                    auk.append([0,1,2])
                if jiji[i][j] == 12:
                    auk.append([0,2])
                if jiji[i][j] == 13:
                    auk.append([0,1,2])
                if jiji[i][j] == 14:
                    auk.append([0])
            for aa in range(len(auk[0])):
                a=auk[0][aa]
                for bb in range(len(auk[1])):
                    b=auk[1][bb]
                    for cc in range(len(auk[2])):
                        c=auk[2][cc]
                        for dd in range(len(auk[3])):
                            d=auk[3][dd]
                            for ee in range(len(auk[4])):
                                e=auk[4][ee]
                                for ff in range(len(auk[5])):
                                    f=auk[5][ff]
                                    for gg in range(len(auk[6])):
                                        g=auk[6][gg]
                                        for hh in range(len(auk[7])):
                                            h=auk[7][hh]
                                            konboberri=[a,b,c,d,e,f,g,h,i]
                                            self.konbos[kont].append(konboberri)     
            kont+=1                                        
prueba = Baseelige()
prueba.konbuk()
lista=np.array(prueba.konbos, dtype=object)
np.save("combinations/kopis",lista)
#listaBuelta=np.load("motza.npy")
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
