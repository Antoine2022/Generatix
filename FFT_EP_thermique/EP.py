import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.integrate as integrate
import scipy.special as special
import numba
from numba import jit
import random
import csv
import time
from microstructures import *
from greenOperatorConductionNumba import *
from Reference import *

def ka(E,nu):
    return E/3/(1-2*nu)

def mu(E,nu):
    return E/2/(1+nu)

def E_(k_oed,mu_cis):
    return 9*k_oed*mu_cis/(mu_cis+3*k_oed)

def nu_(k_oed,mu_cis):
    return (3*k_oed-2*mu_cis)/(2*mu_cis+6*k_oed)

@jit
def nullifyField(field,N):
    for n in np.ndindex(N):
        field[n] = np.zeros(len(N))

@jit
def rec_fill(t,N):
    d=len(N)
    if t[0].shape==(d,d):
        for ii in prange(N[0]):
            t[ii]=np.eye(d)
    else:
        for ii in prange(N[0]):
            rec_fill(t[ii],N)
    

def getGreenMatrix(M,k0,gammaLocal,shifts):
    d=len(M)
    numDOF = d*2**d
    matrix = np.zeros((numDOF,numDOF))
    i = 0
    for I in np.ndindex(shifts):
        j = 0
        for J in np.ndindex(shifts):
            shift=tuple(np.array(I)-np.array(J))
            matrix[d*i:d*(i+1),d*j:d*(j+1)] -= gammaLocal[shift]
            j+=1
        i+=1
    return matrix

def Gamma(tau_field,N,k0):
    d=len(N)
    filter_level=2
    Eps_f=np.zeros(tuple(N)+(d,d))
    for j in range(d):
        tau = np.zeros(tuple(N)+(d,))
        for i in np.ndindex(tuple(N)):
            tau[i]=tau_field[i][:,j]
        field,field_fourier,fft,ifft,tupleK,frequencies,filters,tupleFilters = initializeGreen(N,filter_level=filter_level)
        nullifyField(field,tuple(N))
        Eps_field = np.zeros(tuple(N)+(d,))
        for i in np.ndindex(tuple(N)):
            Eps_field[i+(j,)]=0.
        for i in np.ndindex(tuple(N)+(d,)):
            field[i] = tau[i]
        field=operate_field(field,field_fourier,fft,ifft,tupleK,N,frequencies,filter_level,filters,tupleFilters,k0)
        for i in np.ndindex(tuple(N)):
            Eps_field[i]-=field[i]
        for i in np.ndindex(tuple(N)+(d,)):
            Eps_f[i+(j,)]=Eps_field[i]
    return Eps_f

N0=128
#for frac0 in [0.001,0.005,0.0075,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.12,0.15,0.2]:
#for frac0 in [0.01,0.03,0.07,0.15]:
for frac0 in [0.3]:#[0.05,0.1,0.15,0.2,0.25,0.3]:#,0.4,0.45,0.55,0.65,0.8,1,1.25,1.5,2,2.8,3.5]:# [0.01,0.1,0.2,0.3,0.4,0.5,0.7,1,1.5,2.5,3,4]:
#frac=0.01
    ratio=0.06
    aspect=1
    #phase1=genere_hard_spheres(N,0.07,frac)
    print('generate micro...')
    #khi,frac=genere_hard_spheres(N0,ratio,frac0)
    khi0=np.load("microstructures/micro_hard_sph_exc=0.01_ratio=0.06_frac="+str(frac0)+".npy")
    print(khi0.shape)
    khi=np.zeros((N0,N0,N0))
    frac=0
    for i in np.ndindex((N0,N0,N0)):
    	i1,j1,k1=i
    	khi[i]=khi0[k1+N0*(j1+N0*i1)]
    	if khi[i]==1:
    	    frac+=1/N0**3
    #plt.figure()
    #plt.imshow(khi[0,:,:],origin='lower')
    #plt.show()
    #plt.close()
    #np.save("./microstructures/micro_spheres_frac="+str(frac)+".npy",khi)
    print(frac)

    d=3
    c_i=2
    l0=(c_i-1)
    N=N0*np.ones(d,dtype=np.int64)
    k0=np.eye(len(N))
    Eb=np.array([1.,0.,0.])
    Jb=np.array([1.,0.,0.])

    tau=np.zeros(tuple(N)+(d,d))
    for i in np.ndindex(tuple(N)):
        tau[i]=khi[i]*np.eye(d)
    
    nb=2
    H=np.zeros((nb,)+tuple(N)+(d,d))
    T_=np.zeros((nb,)+tuple(N)+(d,d))
    k=0
    T=tau
    t0=time.time()
    while(k<nb):
        print('compute G'+str(k))
        H[k]=Gamma(T,N,k0)
        T_[k]=T
        for i in np.ndindex(tuple(N)):
            T[i]=l0*khi[i]*H[k][i]
        k+=1

    print('compute UB')
    list_G=[[np.zeros((d,d)) for _ in range(nb)] for __ in range(nb)]
    list_H=[np.zeros((d,d)) for _ in range(nb)]
    cv=0.
    for i in np.ndindex(tuple(N)):
        sig=(1+l0*khi[i])
        sig=khi[i]
        for k in range(nb):
            list_H[k]+=sig*(H[k][i]).transpose()/N0**d
            for l in range(k,nb):
                list_G[k][l]+=sig*np.dot((H[k][i]).transpose(),H[l][i])/N0**d
        cv+=sig/N0**d
    for k in range(nb):
        for l in range(k):
            list_G[k][l]=np.transpose(list_G[l][k])

    MAT=np.zeros((nb*d,nb*d))
    U=np.zeros((nb*d,))
    for k in range(nb):
        U[k*d:(k+1)*d]=np.dot(list_H[k],Eb)
        print("voilà ",k," ",np.dot(list_H[k],Eb))
        for l in range(nb):
            MAT[k*d:(k+1)*d,l*d:(l+1)*d]=list_G[k][l]
        print("voilà encore ",k," ",(list_G[k][k])[0,0])
    
    sig_UB=np.zeros((nb,))
    print('time',time.time()-t0)
    for k in range(nb):
        tau_=np.dot(np.linalg.inv(MAT[:(k+1)*d,:(k+1)*d]),U[:(k+1)*d])
        sig_UB[k]=cv-np.dot(tau_[:(k+1)*d],U[:(k+1)*d])
        print(sig_UB[k])
    sigma_UB=sig_UB[nb-1]


    print('compute HS LB')
    HS=np.zeros((d,d))
    UHS=np.zeros((d,))
    for i in np.ndindex(tuple(N)):
        HS+=(1/l0*np.eye(d)+H[0][i])*khi[i]/N0**d
        UHS+=khi[i]*Eb/N0**d

    tau_HS=np.dot(np.linalg.inv(HS),UHS)
    sigma_HS_LB=np.dot(Eb,Eb)+frac*np.dot(tau_HS,Eb)
    print(sigma_HS_LB)

    print('compute HS UB')
    tau_par=np.zeros(tuple(N)+(d,d))
    for i in np.ndindex(tuple(N)):
        tau_par[i]=(1-khi[i])*np.eye(d)
    k1=c_i*np.eye(len(N))
    H_HS_UB=Gamma(tau_par,N,k1)

    l02=-l0
    HS=np.zeros((d,d))
    UHS=np.zeros((d,))
    for i in np.ndindex(tuple(N)):
        HS+=(1/l02*np.eye(d)+H_HS_UB[i])*(1-khi[i])/N0**d
        UHS+=(1-khi[i])*Eb/N0**d

    tau_HS_UB=np.dot(np.linalg.inv(HS),UHS)
    sigma_HS_UB=c_i*np.dot(Eb,Eb)+(1-frac)*np.dot(tau_HS_UB,Eb)
    print(sigma_HS_UB)



    F=np.zeros((nb,)+tuple(N)+(d,d))
    for k in range(nb):
        print('compute F'+str(k))
        T_m=0
        for i in np.ndindex(tuple(N)):
            T_m+=T_[k][i]/N0**d
        for i in np.ndindex(tuple(N)):
            F[k][i]=T_[k][i]-T_m*np.eye(d)-H[k][i]
        
    
    print('compute LB...')
    list_G=[[np.zeros((d,d)) for _ in range(nb)] for __ in range(nb)]
    list_H=[np.zeros((d,d)) for _ in range(nb)]
    rr=0.
    for i in np.ndindex(tuple(N)):
        rho=1/(1+l0*khi[i])
        for k in range(nb):
            list_H[k]+=rho*(F[k][i]).transpose()/N0**d
            for l in range(k,nb):
                list_G[k][l]+=rho*np.dot((F[k][i]).transpose(),F[l][i])/N0**d
        rr+=rho/N0**d
    for k in range(nb):
        for l in range(k):
            list_G[k][l]=np.transpose(list_G[l][k])
    
    MAT=np.zeros((nb*d,nb*d))
    U=np.zeros((nb*d,))
    for k in range(nb):
        U[k*d:(k+1)*d]=np.dot(list_H[k],Jb)
        for l in range(nb):
            MAT[k*d:(k+1)*d,l*d:(l+1)*d]=list_G[k][l]
    
    sig_LB=np.zeros((nb,))
    print('time',time.time()-t0)
    for k in range(nb):
        tau_=np.dot(np.linalg.inv(MAT[:(k+1)*d,:(k+1)*d]),U[:(k+1)*d])
        sig_LB[k]=1/(rr-np.dot(tau_[:(k+1)*d],U[:(k+1)*d]))
        print(sig_LB[k])
    
    #sig_f=np.zeros(tuple(N)+(d,))
    #for i in np.ndindex(tuple(N)):
    #    sig_f[i]=Jb-np.dot(F1[i],tau_[:d])-np.dot(F2[i],tau_[d:2*d])-np.dot(F3[i],tau_[2*d:3*d])-np.dot(F4[i],tau_[3*d:4*d])
    #    energieUB+=1/(1+l0*khi[i])*np.dot(sig_f[i],sig_f[i])/N0**d
    #print('energie',1/energieUB)
    #print('time',time.time()-t0)

    #print('compute reference...')
    #kref,energieref,eps_ref=Ref(tau,N,(c_i+1)/2*np.eye(d),sigma)
    #ref=kref[0,0]
    #print(ref)
    #print(2*energieref)
    # eps_delta=np.zeros(tuple(N)+(d,))
    # for i in np.ndindex(tuple(N)):
    #     eps_delta[i]=eps_f[i]-eps_ref[i]
    # plt.figure()
    # plt.imshow(eps_delta[:,:,0],origin='lower')
    # plt.show()
    # plt.close()
    # def H(tau):
    #     return a0+np.dot(vec_h,tau)-0.5*np.dot(tau,np.dot(tens_D,tau))
    # print(H(tau_))

    #file=open('./data/spheres_penetrables_c='+str(c_i)+'_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
    #file=open('./data/spheres_hard_c='+str(c_i)+'_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
    sigma_LB=sig_LB[nb-1]
    #file.write(str(frac)+' '+str(sigma_HS_LB)+' '+str(sigma_HS_UB))
    #for k in range(nb):
    #    file.write(' '+str(sig_LB[k])+' '+str(sig_UB[k]))
    #file.write('\n')
    #file.flush()
    #file=open('../data/c1000/ellipsoides_orientees_HS0_aspect='+str(aspect)+'_penetrables_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
    #file.write(str(frac)+' '+str(HS0)+'\n')
    #file=open('../data/FFT/c1000/reference_ellipsoides_orientees_aspect='+str(aspect)+'_penetrables_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
    #file.write(str(frac)+' '+str(ref)+'\n')
