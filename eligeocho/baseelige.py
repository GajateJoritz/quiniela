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
        a=2
        while a!=3:
            b=2
            while b!=3:
                c=0
                while c!=1:
                    d=2
                    while d!=3:
                        e=0
                        while e!=1:
                            f=0
                            while f!=3:
                                g=0
                                while g!=3:
                                    h=0
                                    while h!=3:
                                        konboberri=[a,b,c,d,e,f,g,h]
                                        self.konbos.append(konboberri)               
                                        h+=1
                                    g+=1
                                f+=1
                            e+=1
                        d+=1
                    c+=1
                b+=1
            a+=1
prueba = Baseelige()
prueba.konbuk()
lista=np.array(prueba.konbos)
np.save("eligemotza",lista)
#listaBuelta=np.load("motza.npy")

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
