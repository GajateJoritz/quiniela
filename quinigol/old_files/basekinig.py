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
            while b!=4:
                c=0
                while c!=8:
                    d=0
                    while d!=12:
                        e=4
                        while e!=16:
                            f=0
                            while f!=16:
                                konboberri=[a,b,c,d,e,f]
                                self.konbos.append(konboberri)
                                f+=1
                            e+=1
                        d+=1
                    c+=1
                b+=1
            a+=1
prueba = Basekinig()
prueba.konbuk()
lista=np.array(prueba.konbos)
np.save("combinations/prueba",lista)
#listaBuelta=np.load("motza.npy")

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

