import numpy as np
from numba import jit, float64, int64, prange, generated_jit
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

from greenOperatorConductionNumba import *

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

#La fonction Ref est un algorithme de type Moulinec et Suquet qui permet de calculer la conductivit  homog n is e (kref)   partir d'une 'image' (ki, qui est un tableau contenant les conductivit s de chaque pixel de l'image). N est la taille de l'image. tau_field est le champ de polarisation qui sert de point de d part de l'algorithme. Et k0 est la conductivit  de r f rence utilis e pour le calcul.
def Ref(tau_field,N,k0,ki):
    d=len(N)
    filter_level=2
    Eps_f=np.zeros(tuple(N)+(d,d))
    for j in range(1):
        tau = np.zeros(tuple(N)+(d,))
        for i in np.ndindex(tuple(N)):
            tau[i]=tau_field[i][:,j]
        it=0
        err=1.
        err2=1.
        preverr=1.
        preverr2=1.
        krefj=k0[:,j]
        while err>1e-5 or err2>0.00001 or preverr>1e-5 or preverr2>1e-5:
            field,field_fourier,fft,ifft,tupleK,frequencies,filters,tupleFilters = initializeGreen(N,filter_level=filter_level)
            nullifyField(field,tuple(N))
            Eps_field = np.zeros(tuple(N)+(d,))
            for i in np.ndindex(tuple(N)):
                Eps_field[i+(j,)]=1.
            for i in np.ndindex(tuple(N)+(d,)):
                field[i] = tau[i]
            field=operate_field(field,field_fourier,fft,ifft,tupleK,N,frequencies,filter_level,filters,tupleFilters,k0)
            for i in np.ndindex(tuple(N)):
                Eps_field[i]+=field[i]
            for i in np.ndindex(tuple(N)):
                tau[i]=np.dot(ki[i]-k0,Eps_field[i])
            it+=1
            prevkrefjj=krefj[j]
            krefj=np.zeros((d,))
            energie=0.
            for i in np.ndindex(tuple(N)):
                krefj+=np.dot(ki[i],Eps_field[i])/N[0]**d
                energie+=0.5*np.dot(np.dot(ki[i],Eps_field[i]),Eps_field[i])/N[0]**d
            preverr=err
            preverr2=err2
            err=np.abs((krefj[j]-prevkrefjj)/prevkrefjj)
            err2=np.abs(krefj[j]-prevkrefjj)
            if (it/10==int(it/10) or it==2 or it==4 or it==6):
                print(it,krefj,err,err2,2*energie)
            if err<=1e-5 and err2<=0.0001 and preverr<=1e-5 or preverr2<=1e-5:
                print(it,krefj,err,err2,preverr,preverr2,2*energie)
        for i in np.ndindex(tuple(N)+(d,)):
            Eps_f[i+(j,)]=Eps_field[i]
        #plt.figure()
        #plt.imshow(Eps_field[:,:,0],origin='lower')
        #plt.show()
        #plt.close()
    kref=np.zeros((d,d))
    for i in np.ndindex(tuple(N)):
        kref+=np.dot(ki[i],Eps_f[i])/N[0]**d
    return kref,energie,Eps_field



