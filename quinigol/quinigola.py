import time
from datetime import timedelta
start_time = time.monotonic()
def faltadinak(bat,bi):
    for z in range(0,4):    
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break

def faltadin(bat,bi):
    for z in range(0,3):
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break
def faltad(bat,bi):
    for z in range(0,2):
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break
def falt(bat,bi):
    for z in range(0,5):
        for y in range(len(bat)):     
            if bat[y]==bi[z]:
                del bat[y]
                break
print("iep")
pa1=21.9
pa2=48.2
pa3=21.7
pa4=8.2
pb1=20.6
pb2=37.2
pb3=31.2
pb4=11.0
pc1=6.6
pc2=15.1
pc3=23.5
pc4=54.8
pd1=52.7
pd2=33.5
pd3=9.1
pd4=4.7
pe1=9.8
pe2=28.5
pe3=41.7
pe4=20.0
pf1=38.5
pf2=44.0
pf3=12.7
pf4=4.8
pg1=17.2
pg2=44.4
pg3=30.8
pg4=7.6
ph1=19.6
ph2=42.2
ph3=27.7
ph4=10.5
pi1=23.4
pi2=39.1
pi3=30.0
pi4=7.5
pj1=22.7
pj2=43.8
pj3=26.1
pj4=7.4
pk1=29.1
pk2=45.6
pk3=19.1
pk4=6.2
pl1=14.8
pl2=35.0
pl3=35.2
pl4=15.0
pr1=8
pr2=21
pr3=26
pr4=45
ps1=54
ps2=33
ps3=10
ps4=3
pt1=19
pt2=32
pt3=26
pt4=23
pu1=36
pu2=37
pu3=19
pu4=8
pv1=11
pv2=25
pv3=27
pv4=37
pw1=46
pw2=36
pw3=14
pw4=4
px1=31
px2=36
px3=21
px4=12
py1=25
py2=35
py3=24
py4=16
pz1=14
pz2=28
pz3=27
pz4=31
po1=42
po2=36
po3=16
po4=6
pp1=26
pp2=51
pp3=17
pp4=6
pq1=8
pq2=20
pq3=38
pq4=34
#16.777.216 konbinazixo
"bene1 = [pr1*ps1/10000,pr1*ps2/10000,pr1*ps3/10000,pr1*ps4/10000,pr2*ps1/10000,pr2*ps2/10000,pr2*ps3/10000,pr2*ps4/10000,pr3*ps1/10000,pr3*ps2/10000,pr3*ps3/10000,pr3*ps4/10000,pr4*ps1/10000,pr4*ps2/10000,pr4*ps3/10000,pr4*ps4/10000]"
bene1=[0.08028884574377096, 0.09445746558090701, 0.06690737145314246, 0.03823278368750998, 0.08920982860418995, 0.12352130114426302, 0.08028884574377096, 0.05352589716251398, 0.06176065057213151, 0.08028884574377096, 0.05352589716251398, 0.03823278368750998, 0.03490819380163955, 0.04225728723356366, 0.03490819380163955, 0.027685808877162398]
bene2=[0.04662586916542904, 0.034462598948360595, 0.01933267745883643, 0.005249270038492011, 0.08807108620136596, 0.07926397758122937, 0.034462598948360595, 0.011163940504398503, 0.11323425368747053, 0.08807108620136596, 0.04171788293748914, 0.015541956388476346, 0.17614217240273192, 0.14411632287496248, 0.07205816143748124, 0.030486145223549758]
bene3= [0.08737792283465196, 0.07149102777380614, 0.037447681214850835, 0.015419633441409168, 0.12098481623259502, 0.12098481623259502, 0.05242675370079117, 0.02184448070866299, 0.09830016318898344, 0.09251780064845501, 0.046258900324227505, 0.017095680554605818, 0.07864013055118675, 0.07864013055118675, 0.041389542395361446, 0.019180519646630916]
bene4=[0.09766711201947516, 0.09766711201947516, 0.06010283816583087, 0.026942651591579353, 0.1041782528207735, 0.1302228160259669, 0.07103062692325465, 0.037206518864561965, 0.07103062692325465, 0.07813368961558012, 0.05208912641038675, 0.030051419082915434, 0.04596099389151772, 0.04596099389151772, 0.030051419082915434, 0.02170380267099448]
bene5= [0.06042771235799887, 0.07141456915036229, 0.052370684043599014, 0.04134527687652554, 0.07855602606539852, 0.12085542471599774, 0.07855602606539852, 0.06042771235799887, 0.06042771235799887, 0.07855602606539852, 0.06546335505449877, 0.046209427097293254, 0.04134527687652554, 0.06042771235799887, 0.046209427097293254, 0.03740763145971358]
bene6= [0.12000655713760366, 0.1300071035657373, 0.0866714023771582, 0.052002841426294924, 0.09176972016404987, 0.1300071035657373, 0.08210974962046567, 0.052002841426294924, 0.045884860082024935, 0.06000327856880183, 0.045884860082024935, 0.030001639284400916, 0.0190254297901079, 0.02166785059428955, 0.0190254297901079, 0.013929332524900426]
lae1 = [pa1*pb1,pa1*pb2,pa1*pb3,pa1*pb4,pa2*pb1,pa2*pb2,pa2*pb3,pa2*pb4,pa3*pb1,pa3*pb2,pa3*pb3,pa3*pb4,pa4*pb1,pa4*pb2,pa4*pb3,pa4*pb4]
"bene2 = [pt1*pu1/10000,pt1*pu2/10000,pt1*pu3/10000,pt1*pu4/10000,pt2*pu1/10000,pt2*pu2/10000,pt2*pu3/10000,pt2*pu4/10000,pt3*pu1/10000,pt3*pu2/10000,pt3*pu3/10000,pt3*pu4/10000,pt4*pu1/10000,pt4*pu2/10000,pt4*pu3/10000,pt4*pu4/10000]"
lae2 = [pc1*pd1,pc1*pd2,pc1*pd3,pc1*pd4,pc2*pd1,pc2*pd2,pc2*pd3,pc2*pd4,pc3*pd1,pc3*pd2,pc3*pd3,pc3*pd4,pc4*pd1,pc4*pd2,pc4*pd3,pc4*pd4]
"bene3 = [pv1*pw1/10000,pv1*pw2/10000,pv1*pw3/10000,pv1*pw4/10000,pv2*pw1/10000,pv2*pw2/10000,pv2*pw3/10000,pv2*pw4/10000,pv3*pw1/10000,pv3*pw2/10000,pv3*pw3/10000,pv3*pw4/10000,pv4*pw1/10000,pv4*pw2/10000,pv4*pw3/10000,pv4*pw4/10000]"
lae3 = [pe1*pf1,pe1*pf2,pe1*pf3,pe1*pf4,pe2*pf1,pe2*pf2,pe2*pf3,pe2*pf4,pe3*pf1,pe3*pf2,pe3*pf3,pe3*pf4,pe4*pf1,pe4*pf2,pe4*pf3,pe4*pf4]
"bene4 = [px1*py1/10000,px1*py2/10000,px1*py3/10000,px1*py4/10000,px2*py1/10000,px2*py2/10000,px2*py3/10000,px2*py4/10000,px3*py1/10000,px3*py2/10000,px3*py3/10000,px3*py4/10000,px4*py1/10000,px4*py2/10000,px4*py3/10000,px4*py4/10000]"
lae4 = [pg1*ph1,pg1*ph2,pg1*ph3,pg1*ph4,pg2*ph1,pg2*ph2,pg2*ph3,pg2*ph4,pg3*ph1,pg3*ph2,pg3*ph3,pg3*ph4,pg4*ph1,pg4*ph2,pg4*ph3,pg4*ph4]
"bene5 = [pz1*po1/10000,pz1*po2/10000,pz1*po3/10000,pz1*po4/10000,pz2*po1/10000,pz2*po2/10000,pz2*po3/10000,pz2*po4/10000,pz3*po1/10000,pz3*po2/10000,pz3*po3/10000,pz3*po4/10000,pz4*po1/10000,pz4*po2/10000,pz4*po3/10000,pz4*po4/10000]"
lae5 = [pi1*pj1,pi1*pj2,pi1*pj3,pi1*pj4,pi2*pj1,pi2*pj2,pi2*pj3,pi2*pj4,pi3*pj1,pi3*pj2,pi3*pj3,pi3*pj4,pi4*pj1,pi4*pj2,pi4*pj3,pi4*pj4]
"bene6 = [pp1*pq1/10000,pp1*pq2/10000,pp1*pq3/10000,pp1*pq4/10000,pp2*pq1/10000,pp2*pq2/10000,pp2*pq3/10000,pp2*pq4/10000,pp3*pq1/10000,pp3*pq2/10000,pp3*pq3/10000,pp3*pq4/10000,pp4*pq1/10000,pp4*pq2/10000,pp4*pq3/10000,pp4*pq4/10000]"
lae6 = [pk1*pl1,pk1*pl2,pk1*pl3,pk1*pl4,pk2*pl1,pk2*pl2,pk2*pl3,pk2*pl4,pk3*pl1,pk3*pl2,pk3*pl3,pk3*pl4,pk4*pl1,pk4*pl2,pk4*pl3,pk4*pl4]
estimazixue = 100000
plenue=["0-0","0-1","0-2","0-M","1-0","1-1","1-2","1-M","2-0","2-1","2-2","2-M","M-0","M-1","M-2","M-M"]
portzentaj=0.0
portzentaj2=0.0
konkretu=[0,0,0,0,0,0]
seikue=0.0
bostekue=0.0
laukue=0.0
hirukue=0.0
bikue=0.0
jeje=0.0
from itertools import combinations
irabazixek2=[]
irabazixek3=[]
irabazixek4=[]
irabazixek5=[]
irabazixek6=[]
for ab in range(0,estimazixue+1):
    irabazixek2.append(0)
    irabazixek3.append(0)
    irabazixek4.append(0)
    irabazixek5.append(0)
    irabazixek6.append(0)
for yi in range(0,estimazixue+1):
    if (estimazixue*0.08)/(yi+1)>40000:
        irabazixek4[yi]=((estimazixue*0.08)/(yi+1)-40000)*0.8+40000
        irabazixek3[yi]=((estimazixue*0.08)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek4[yi]=(estimazixue*0.08)/(yi+1)
        irabazixek3[yi]=(estimazixue*0.08)/(yi+1)
    if (estimazixue*0.09)/(yi+1)>40000:
        irabazixek5[yi]=((estimazixue*0.09)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek5[yi]=(estimazixue*0.09)/(yi+1)
    if (estimazixue*0.1+0)/(yi+1)>40000:
        irabazixek6[yi]=((estimazixue*0.1+0)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek6[yi]=(estimazixue*0.1+0)/(yi+1)
    if (estimazixue*0.2)/(yi+1)>40000:
        irabazixek2[yi]=((estimazixue*0.2)/(yi+1)-40000)*0.8+40000
    else:
        irabazixek2[yi]=(estimazixue*0.2)/(yi+1)
sobrak=[]
sobrak2=[]
sobrak3=[]
sobrak4=[]
sobrak5=[]
sobrak6=[]
sobrak7=[]
sobrak8=[]
for a in range(0,1):
    for b in range(12,13):
        for c in range(4,5):
            for d in range(0,1):
                for e in range(15,16):
                    for f in range(0,1):
                        benetakue6=(bene1[a])*(bene2[b])*(bene3[c])*(bene4[d])*(bene5[e])*(bene6[f])
                        laekue6=0.0
                        laekue5=0.0
                        laekue4=0.0
                        laekue3=0.0
                        laekue2=0.0
                        laek4=0.0
                        laek3=0.0
                        laek2=0.0    
                        benetakue5=0.0      
                        benetakue4ogexo=0.0
                        benetakue4=0.0
                        benetakue3=0.0
                        benetakue2=0.0
                        bene=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f])]
                        lae=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f])]
                        konbin=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5]], 5)))
                        konbin2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5]], 5)))
                        konbinazixuk=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5]], 4)))
                        konbinazixuk2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5]], 4)))
                        konbinazix=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5]], 3)))
                        konbinazix2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5]], 3)))
                        #konbinaz=(list(combinations([bene[0],bene[1],bene[2],bene[3],bene[4],bene[5]], 2)))
                        #konbinaz2=(list(combinations([lae[0],lae[1],lae[2],lae[3],lae[4],lae[5]], 2)))
                        for zu in range(0,6):
                            sobrak7=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f])]
                            sobrak8=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f])]
                            falt(sobrak7,konbin[zu])                                                       
                            falt(sobrak8,konbin2[zu])
                            benetakue5=benetakue5+(konbin[zu][0]/100)*(konbin[zu][1]/100)*(konbin[zu][2]/100)*(konbin[zu][3]/100)*(konbin[zu][4]/100)*(1-sobrak7[0]/100)
                        for z in range(0,15):
                            sobrak=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f])]
                            sobrak2=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f])]       
                            faltadinak(sobrak,konbinazixuk[z])
                            faltadinak(sobrak2,konbinazixuk2[z])
                            benetakue4=benetakue4+konbinazixuk[z][0]*konbinazixuk[z][1]*konbinazixuk[z][2]*konbinazixuk[z][3]*(1-sobrak[0])*(1-sobrak[1])
                            benetakue4ogexo=benetakue4ogexo+konbinazixuk[z][0]*konbinazixuk[z][1]*konbinazixuk[z][2]*konbinazixuk[z][3]
                            laek4=laek4+(konbinazixuk2[z][0]/10000)*(konbinazixuk2[z][1]/10000)*(konbinazixuk2[z][2]/10000)*(konbinazixuk2[z][3]/10000)*(1-sobrak2[0]/10000)*(1-sobrak2[1]/10000)
                        for zi in range(0,20):
                            sobrak3=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f])]
                            sobrak4=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f])]       
                            faltadin(sobrak3,konbinazix[zi])
                            faltadin(sobrak4,konbinazix2[zi])
                            benetakue3=benetakue3+konbinazix[zi][0]*konbinazix[zi][1]*konbinazix[zi][2]*(1-sobrak3[0])*(1-sobrak3[1])*(1-sobrak3[2])
                            laek3=laek3+(konbinazix2[zi][0]/10000)*(konbinazix2[zi][1]/10000)*(konbinazix2[zi][2]/10000)*(1-sobrak4[0]/10000)*(1-sobrak4[1]/10000)*(1-sobrak4[2]/10000)
                        #for zo in range(0,15):
                        #    sobrak5=[(bene1[a]),(bene2[b]),(bene3[c]),(bene4[d]),(bene5[e]),(bene6[f])]
                        #    sobrak6=[(lae1[a]),(lae2[b]),(lae3[c]),(lae4[d]),(lae5[e]),(lae6[f])]     
                        #    faltad(sobrak5,konbinaz[zo])
                        #    faltad(sobrak6,konbinaz2[zo])
                        #    benetakue2=benetakue2+konbinaz[zo][0]*konbinaz[zo][1]*(1-sobrak5[0])*(1-sobrak5[1])*(1-sobrak5[2])*(1-sobrak5[3])
                        #    laek2=laek2+(konbinaz2[zo][0]/10000)*(konbinaz2[zo][1]/10000)*(1-sobrak6[0]/10000)*(1-sobrak6[1]/10000)*(1-sobrak6[2]/10000)*(1-sobrak6[3]/10000)
                        for y in range(0,5):
                            laekue2=laekue2+((1-(laek2))**(estimazixue-y))*irabazixek2[y]*(laek2**y)
                            laekue3=laekue3+((1-(laek3))**(estimazixue-y))*irabazixek3[y]*(laek3**y)
                            laekue4=laekue4+((1-(laek4))**(estimazixue-y))*irabazixek4[y]*(laek4**y)
                            laekue5=laekue5+(((1-((lae1[a]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae6[f]/10000))+(lae1[a]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae6[f]/10000)*(1-(lae5[e])/10000)+(lae1[a]/10000)*(lae2[b]/10000)*(lae3[c])/10000*(lae6[f]/10000)*(lae5[e]/10000)*(1-(lae4[d])/10000)+(lae1[a]/10000)*(lae2[b]/10000)*(lae6[f]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae3[c])/10000)+(lae1[a]/10000)*(lae6[f]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae2[b])/10000)+(lae6[f]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae1[a])/10000)))**(estimazixue-y))*irabazixek5[y]*(((lae1[a]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae6[f]/10000))+(lae1[a]/10000)*(lae2[b])/10000*(lae3[c]/10000)*(lae4[d]/10000)*(lae6[f]/10000)*(1-(lae5[e])/10000)+(lae1[a]/10000)*(lae2[b]/10000)*(lae3[c])/10000*(lae6[f]/10000)*(lae5[e]/10000)*(1-(lae4[d])/10000)+(lae1[a]/10000)*(lae2[b]/10000)*(lae6[f]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae3[c])/10000)+(lae1[a]/10000)*(lae6[f]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae2[b])/10000)+(lae6[f]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(1-(lae1[a])/10000)))**y)
                            laekue6=laekue6+((1-(lae1[a]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(lae6[f]/10000))**(estimazixue-y))*irabazixek6[y]*(((lae1[a]/10000)*(lae2[b]/10000)*(lae3[c]/10000)*(lae4[d]/10000)*(lae5[e]/10000)*(lae6[f]/10000))**y)                     
                        maxi = (benetakue6*laekue6)+(benetakue5*laekue5)+(benetakue4*laekue4)+(benetakue3*laekue3)+(benetakue2*laekue2)
                        if benetakue4ogexo>0.00:
                            if maxi>portzentaj:
                                portzentaj2=portzentaj
                                portzentaj=maxi
                                seikue2=seikue
                                seikue=benetakue6*laekue6
                                bostekue2=bostekue
                                bostekue=benetakue5*laekue5
                                laukue2=laukue
                                laukue=benetakue4*laekue4
                                hirukue2=hirukue
                                hirukue=benetakue3*laekue3
                                bikue2=bikue
                                bikue=benetakue2*laekue2
                                konkretu2=konkretu
                                konkretu=[a,b,c,d,e,f]
                                jeje2=jeje
                                jeje=benetakue4ogexo
                            elif maxi>portzentaj2:
                                portzentaj2=maxi
                                seikue2=benetakue6*laekue6
                                bostekue2=benetakue5*laekue5
                                laukue2=benetakue4*laekue4
                                hirukue2=benetakue3*laekue3
                                bikue2=benetakue2*laekue2
                                konkretu2=[a,b,c,d,e,f]
                                jeje2=benetakue4ogexo
print(portzentaj)
print(plenue[konkretu[0]],plenue[konkretu[1]],plenue[konkretu[2]],plenue[konkretu[3]],plenue[konkretu[4]],plenue[konkretu[5]])
print("Seikuena:")
print(seikue)
print("Bostekuena:")
print(bostekue)
print("Laukuena:")
print(laukue)
print("Hirukuena:")
print(hirukue)
print("Bikuena:")
print(bikue)
print("jeje:")
print(jeje)
print()
print(portzentaj2)
print(plenue[konkretu2[0]],plenue[konkretu2[1]],plenue[konkretu2[2]],plenue[konkretu2[3]],plenue[konkretu2[4]],plenue[konkretu2[5]])
print("Seikuena:")
print(seikue2)
print("Bostekuena:")
print(bostekue2)
print("Laukuena:")
print(laukue2)
print("Hirukuena:")
print(hirukue2)
print("Bikuena:")
print(bikue2)
print("jeje:")
print(jeje2)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))    