import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import scipy.special as special
from random import uniform
from numba import jit


def genere_aleat(N,frac):
    N1=int(frac*N**3)
    phase1=np.zeros((N1,3))
    for i in range(1,N1):
        if i/20==int(i/20):
            print(i)
        bool=True
        while bool:
            element=int(N*uniform(0,1)),int(N*uniform(0,1)),int(N*uniform(0,1))
            j=0
            while j<i:
                el=phase1[j]
                if (el==element).all():
                    j=i+1
                else:
                    j+=1
            if j==i:
                bool=False
                phase1[i]=element
    print(len(phase1),phase1[len(phase1)-1])
    return phase1

#@jit(nopython=True)
def genere_hard_spheres(N,ratio,frac):
    r=ratio*N
    Vol=4/3*np.pi*r**3
    Voltot=N**3
    nb=int(frac*Voltot/Vol)
    print(nb,Vol,Voltot)

    def intersecte(c1,c2):
        bool=False
        x1,y1,z1=c1
        x2,y2,z2=c2
        for ii in [-N,0,N]:
            for jj in [-N,0,N]:
                for kk in [-N,0,N]:
                    if (x1+ii-x2)**2+(y1+jj-y2)**2+(z1+kk-z2)**2<r**2:
                        bool=True
        return bool

    centers=[[0,0,0]]
    i=1
    while i<nb:
        c1=[uniform(0,N),uniform(0,N),uniform(0,N)]
        j=0
        bool=False
        #while j<len(centers):
        #    c2=centers[j]
        #    if intersecte(c1,c2):
        #        bool=True
        #        j=len(centers)
        #    else:
        #        j+=1
        if not(bool):
            centers.append(c1)
            #print(c1)
            i+=1
    ph=np.zeros((N,N,N))
    nbr=0
    for center in centers:
        x,y,z=center
        for ii in range(-int(r+2),int(r+2)):
            for jj in range(-int(r+2),int(r+2)):
                for kk in range(-int(r+2),int(r+2)):
                    if (ii)**2+(jj)**2+(kk)**2<r**2:
                        i1=int(x+ii)
                        j1=int(y+jj)
                        k1=int(z+kk)
                        if i1>=N:
                            i1-=N
                        if i1<0:
                            i1+=N
                        if j1>=N:
                            j1-=N
                        if j1<0:
                            j1+=N
                        if k1>=N:
                            k1-=N
                        if k1<0:
                            k1+=N
                        if ph[i1,j1,k1]==0:
                            nbr+=1                     
                        ph[i1,j1,k1]=1
   
    print(frac,'frac reelle',nbr/N**3)
    phase1=np.array(ph)
    return phase1,nbr/N**3

def genere_hard_disks(N,ratio,frac,aspect):
    r=ratio*N
    Vol=np.pi*r**2
    Voltot=N**2
    nb=int(frac*Voltot/Vol)
    #print(nb,Vol,Voltot)
    def intersecte(c1,c2):
        bool=False
        x1,y1=c1
        x2,y2=c2
        for ii in [-N,0,N]:
            for jj in [-N,0,N]:
                if (x1+ii-x2)**2+(y1+jj-y2)**2<0*r**2:
                    bool=True
        return bool

    centers=[[0,0]]
    i=1
    while i<nb:
        c1=[uniform(0,N),uniform(0,N)]
        j=0
        bool=False
        while j<len(centers):
            c2=centers[j]
            if intersecte(c1,c2):
                bool=True
                j=len(centers)
            else:
                j+=1
        if not(bool):
            centers.append(c1)
            #print(c1)
            i+=1
    ph=np.zeros((N,N))
    nbr=0
    for center in centers:
        x,y=center
        for ii in range(-int(aspect*r+2),int(aspect*r+2)):
            for jj in range(-int(r+2),int(r+2)):
                if (ii)**2/aspect**2+(jj)**2<r**2:
                    i1=int(x+ii)
                    j1=int(y+jj)
                    if i1>=N:
                        i1-=N
                    if i1<0:
                        i1+=N
                    if j1>=N:
                        j1-=N
                    if j1<0:
                        j1+=N   
                    if ph[i1,j1]==0:
                        nbr+=1                  
                    ph[i1,j1]=1
    print(frac,'frac reelle',nbr/N**2)
    return ph,nbr/N**2

def genere_spheres(N,ratio,frac):
    r=ratio*N
    Vol=4/3*np.pi*r**3
    Voltot=N**3
    nb=int(frac*Voltot/Vol)
    print(nb,Vol,Voltot)

    def intersecte(c1,c2):
        bool=False
        x1,y1,z1=c1
        x2,y2,z2=c2
        for ii in [-N,0,N]:
            for jj in [-N,0,N]:
                for kk in [-N,0,N]:
                    if (x1+ii-x2)**2+(y1+jj-y2)**2+(z1+kk-z2)**2<r**2:
                        bool=True
        return bool

    centers=[[0,0,0]]
    i=1
    while i<nb:
        c1=[uniform(0,N),uniform(0,N),uniform(0,N)]
        j=0
        bool=False
        while j<len(centers):
            c2=centers[j]
            if intersecte(c1,c2):
                bool=True
                j=len(centers)
            else:
                j+=1
        if not(bool):
            centers.append(c1)
            #print(c1)
            i+=1
    ph=[]
    for center in centers:
        x,y,z=center
        for ii in range(-int(r+2),int(r+2)):
            for jj in range(-int(r+2),int(r+2)):
                for kk in range(-int(r+2),int(r+2)):
                    if (ii)**2+(jj)**2+(kk)**2<r**2:
                        i1=int(x+ii)
                        j1=int(y+jj)
                        k1=int(z+kk)
                        if i1>=N:
                            i1-=N
                        if i1<0:
                            i1+=N
                        if j1>=N:
                            j1-=N
                        if j1<0:
                            j1+=N
                        if k1>=N:
                            k1-=N
                        if k1<0:
                            k1+=N                        
                        ph.append([i1,j1,k1])
    print(len(ph))
    print('frac reelle',len(ph)/N**3)
    phase1=np.array(ph)
    return phase1

    

#Vol=4/3*np.pi*1.5**3

# R=1
# r=0.01
# l=1
# ndr=20
# nb=20
# r_list=np.linspace(2*r,92*r,ndr)
# S_list=np.zeros((ndr,))
# i0_list=np.zeros((ndr,))
# for mm in range(nb):
#     print(mm)
#     m=genere_micro(R,l,r,0.04)
#     normals=m[:,1,:]
#     centers=m[:,0,:]
#     N=len(centers)
#     min=R
#     for alpha in range(N):
#         if dist(centers[alpha],np.array([0,0,0]))<min:
#             min=dist(centers[alpha],np.array([0,0,0]))
#             centre=alpha
#     alpha=centre
#         
#     for i in range(ndr):
#         rr=r_list[i]
#         rrdr=r_list[i]+2*R/ndr
#         S=0.
#         i0=0
#         for beta in range(N):
#             distance=np.linalg.norm(centers[beta]-centers[alpha])
#             if distance>=rr and distance<rrdr:
#                 S+=0.5*(3*(np.dot(normals[alpha],normals[beta]))**2-1)
#                 i0+=1
#         if S!=0:
#             S_list[i]+=S/i0/nb
#             i0_list[i]+=i0/nb
#     
#     # plt.figure()
#     # plt.plot(r_list,S_list)
#     # plt.plot(r_list,i0_list)
#     # plt.show()
#     # plt.close()
#     
# plt.figure()
# plt.plot(r_list,S_list)
# plt.plot(r_list,i0_list)
# plt.show()
# plt.close()
    

# R=1
#m=genere_micro(0.5,1,0.01,0.01)
#N=len(m)
#for i in range(len(m)):
#   print(m[i,0,:],m[i,2,:])
# 
# mbis=np.zeros((2*N,2,3))
# for i in range(len(m)):
#     center1=m[i,0]
#     normal=m[i,1]
#     center2=m[i,0]+np.array([-2*R,0,0])
#     mbis[i,0]=center1
#     mbis[i+N,0]=center2
#     mbis[i,1]=normal
#     mbis[i+N,1]=normal
#             
# m=mbis

# print(normals)
# nn1=0.
# nn2=0.
# nn3=0.
# for i in range(len(normals)):
#     nn1+=normals[i,0]/len(normals)
#     nn2+=normals[i,1]/len(normals)
#     nn3+=normals[i,2]/len(normals)
# print(nn1,nn2,nn3)
#
# 
# a1=np.array([0,0,0])
# b1=np.array([0,0,1])
# a2=np.array([0.56,-0.2,0])
# b2=np.array([0.56,0.2,1])
# print(dist_segment(a1,b1,a2,b2))
# 
# l=1
# r=0.01
# center1=np.array([0,0,0])
# normal1=np.array([0,0,1])
# center2=np.array([0.519,0,0])
# normal2=np.array([1,0,0])
# c1=np.array([center1,normal1])
# c2=np.array([center2,normal2])
# print(test_recouvrement(c1,c2,l,r))
#nb=30
# l=1
# m2=[]
# for i in range(len(m)):
#     center=m[i,0]
#     normal=m[i,1]
#     ll=[]
#     for k in range(nb):
#         pp=center+normal*(-l/2+l*k/(nb-1))
#         ll.append(pp)
#     m2.append(np.array(ll))
# m2=np.array(m2)
# xs=m2[:,:,0]
# xs=np.reshape(xs,(nb*len(xs),))
# ys=m2[:,:,1]
# ys=np.reshape(ys,(nb*len(ys),))
# zs=m2[:,:,2]
# zs=np.reshape(zs,(nb*len(zs),))
# print(zs)

# fig=plt.figure()
# ax=fig.add_subplot(111,projection='3d')
# ax.scatter(np.array([0.]),np.array([0.]),np.array([0.]),c='r',marker='o',s=1.68)
# plt.xlim([-2,2])
# plt.ylim([-2,2])
# ax.set_zlim(-2,2)
# plt.show()
# plt.close()

# fig=plt.figure()
# ax=fig.add_subplot(111,projection='3d')
# ax.scatter(xs,ys,zs,c='r',marker='o',s=1.68)
# plt.show()
# plt.close()
