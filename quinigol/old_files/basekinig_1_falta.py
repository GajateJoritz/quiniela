import time
import pickle
from datetime import timedelta
start_time = time.monotonic()
import numpy as np
class Basekinig():
    def __init__(self):
        self.konbos=[]
    def konbuk(self):
        a=0
        while a!=16:
            b=0
            while b!=16:
                c=0
                while c!=16:
                    d=0
                    while d!=16:
                        e=0
                        while e!=16:
                            konboberri=[a,b,c,d,e]
                            self.konbos.append(konboberri)
                            e+=1
                        d+=1
                    c+=1
                b+=1
            a+=1
prueba = Basekinig()
prueba.konbuk()
lista=np.array(prueba.konbos)
np.save("1_falta_kinigol_konbuk",lista)
#listaBuelta=np.load("motza.npy")

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

