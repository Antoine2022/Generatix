from matplotlib import pyplot as plt
import numpy as np

frac=[]
HLB=[]
HUB=[]
sLB=[]
sUB=[]
sLBe=[]
sUBe=[]
sFFT=[]
c_i=1e2
nb=4
#[('f',float),('LB',float),('UB',float),('LB',float),('UB',float),('LB',float),('UB',float),('LB',float),('UB',float)]
list_type=[('f',float)]#,('HSLB',float),('HSUB',float)]
for k in range(nb):
    list_type.append(tuple(('LB'+str(k),float)))
    list_type.append(tuple(('UB'+str(k),float)))
    sLB.append([])
    sUB.append([])
    sLBe.append([])
    sUBe.append([])
list_type.append(tuple(('FFT',float)))


#file=np.loadtxt('./data/spheres_penetrables_c='+str(c_i)+'_N0=128_ratio=0.06.txt',dtype=list_type)
#file=np.loadtxt('./data/FFT_spheres_penetrables_c='+str(c_i)+'_N0=128_ratio=0.06.txt',dtype=list_type)
file=np.loadtxt('./data/tmfft_spheres_penetrables_c='+str(c_i)+'_N0=128_ratio=0.06_XX.txt',dtype=list_type)
#file=np.loadtxt('./data/spheres_hard_c='+str(c_i)+'_N0=128_ratio=0.06.txt',dtype=list_type)
for i in range(len(file)):
    frac.append(file[i]['f'])
    #HLB.append(file[i]['HSLB']/c_i)
    #HUB.append(file[i]['HSUB']/c_i)
    for k in range(nb):
        sLB[k].append(file[i]['LB'+str(k)]/c_i)
        sUB[k].append(file[i]['UB'+str(k)]/c_i)
        sUBe[k].append((file[i]['UB'+str(k)]-file[i]['UB'+str(nb-1)])/file[i]['UB'+str(nb-1)])
    	
        sLBe[k].append((file[i]['LB'+str(k)]-file[i]['LB'+str(nb-1)])/file[i]['LB'+str(nb-1)])        
    sFFT.append(file[i]['FFT']/c_i)

it=[]
sUBi=[]
for k in range(nb):
	sUBi.append(sUBe[k][5])
	it.append(k+1)

xx=np.linspace(0,1,100)
yy=np.linspace(0,0,100)
for i in range(100):
	f=xx[i]
	yy[i]=(1-f)/c_i+f-f*(1-f)*(c_i-1)**2/c_i/(c_i+2-f*(c_i-1))

plt.figure()
#gH0,=plt.plot(frac,HLB,'+',color='green')
#gH1,=plt.plot(frac,HUB,'+',color='green')
g1,=plt.plot(frac,sLB[0],'+',color='blue')
#g2,=plt.plot(frac,sUB[0],'+',color='blue')
#g1,=plt.plot(frac,sLB[0],'+',color='green')
g2,=plt.plot(frac,sUB[0],'+',color='orange')
#g2,=plt.plot(frac,sUB[1],'+',color='green')
#g2,=plt.plot(frac,sUB[2],'+',color='red')
g2,=plt.plot(frac,sUB[3],'+',color='black')
#g2,=plt.plot(frac,sLB[0],'o',color='blue',markersize=2)
#g2,=plt.plot(frac,sLB[1],'o',color='green',markersize=2)
#g2,=plt.plot(frac,sLB[2],'o',color='red',markersize=2)
g2,=plt.plot(frac,sLB[3],'+',color='green')
g3,=plt.plot(frac,sFFT,'o',markersize=2,color='red')
g2,=plt.plot(xx,yy,'--',color='red')
#g2,=plt.plot(it,sUBi,'+',color='black')
#plt.legend([gH0,gH1,g1,g2],['HS-','HS+','LB','UB'])
#plt.legend([g1,g2],['LB','UB'])
plt.xlabel('f')
plt.ylabel('sig11')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.savefig('1.pdf')
plt.show()
plt.close()
