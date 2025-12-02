import time
import pickle
from datetime import timedelta
start_time = time.monotonic()
import numpy as np
class Base():
    def __init__(self):
        self.konbos=[]
    def konbuk(self):
        a=0
        while a!=3:
            b=0
            while b!=3:
                c=0
                while c!=3:
                    d=0
                    while d!=3:
                        e=0
                        while e!=3:
                            f=0
                            while f!=3:
                                g=0
                                while g!=3:
                                    h=0
                                    while h!=3:
                                        i=0
                                        while i!=3:
                                            j=0
                                            while j!=3:
                                                k=0
                                                while k!=3:
                                                    l=0
                                                    while l!=3:
                                                        m=0
                                                        while m!=3:
                                                            n=0
                                                            while n!=3:
                                                                o=2
                                                                while o!=3:
                                                                    konboberri=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o]
                                                                    self.konbos.append(konboberri)
                                                                    o+=1
                                                                n+=1
                                                            m+=1
                                                        l+=1
                                                    k+=1
                                                j+=1
                                            i+=1
                                        h+=1
                                    g+=1
                                f+=1
                            e+=1
                        d+=1
                    c+=1
                b+=1
            a+=1
prueba = Base()
prueba.konbuk()
lista=np.array(prueba.konbos)
np.save("motza",lista)
#listaBuelta=np.load("motza.npy")

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

