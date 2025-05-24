import numpy as np
import scipy.special as special
from random import uniform
from numba import jit

@jit(nopython=True)
def is_near(p0,pi,d,D):
    value=False
    j1=-1
    while j1<2:
        j2=-1
        while j2<2:
            j3=-1
            while j3<2:
                vec=np.array([j1*D,j2*D,j3*D])
                pi_=pi+vec
                if np.linalg.norm(p0-pi_)<=d:
                    value=True
                    j1=2
                    j2=2
                    j3=2
                else:
                    j3+=1
            j2+=1
        j1+=1
    return value

@jit(nopython=True)
def inside(pc,pi,n,l,e,D):
    value=False
    j1=-1
    while j1<2:
        j2=-1
        while j2<2:
            j3=-1
            while j3<2:
                vec=np.array([j1*D,j2*D,j3*D])
                pi_=pi+vec
                u=pi_-pc
                x=np.dot(u,n)
                if x<l/2:
                    u_n=u-x*n
                    a=l/2
                    b=l/2/e
                    y=np.linalg.norm(u_n)
                    if (x/a)**2+(y/b)**2<=1:
                        value=True
                        j1=2
                        j2=2
                        j3=2
                    else:
                        j3+=1
                else:
                    j3+=1
            j2+=1
        j1+=1
    return value
    

@jit(nopython=True)
def fill(micro_v,c,n,l,e,i,D):
    N0,N1,N2=micro_v.shape
    xc,yc,zc=c
    xc=xc+D/2
    yc=yc+D/2
    zc=zc+D/2
    p1=c-l/2*n+D/2*np.array([1,1,1])
    p2=c+l/2*n+D/2*np.array([1,1,1])
    x1,y1,z1=p1
    x2,y2,z2=p2
    r=l/e/2
    i1=int(x1*N0)
    j1=int(y1*N1)
    k1=int(z1*N2)
    i2=int(x2*N0)
    j2=int(y2*N1)
    k2=int(z2*N2)
    im=min(i1,i2)-int(r*N0)-1
    iM=max(i1,i2)+int(r*N0)+1
    jm=min(j1,j2)-int(r*N1)-1
    jM=max(j1,j2)+int(r*N1)+1
    km=min(k1,k2)-int(r*N2)-1
    kM=max(k1,k2)+int(r*N2)+1
    for ii in range(im,iM):
        for jj in range(jm,jM):
            for kk in range(km,kM):
                if ii>=N0:
                    ii-=N0
                if jj>=N1:
                    jj-=N1
                if kk>=N2:
                    kk-=N2
                xi=ii/N0
                yi=jj/N1
                zi=kk/N2
                pi=np.array([xi,yi,zi])
                pc=np.array([xc,yc,zc])
                #if is_near(pc,pi,l/2,D):
                value=inside(pc,pi,n,l,e,D)
                if value:
                    micro_v[ii,jj,kk]=i


def voxelize_ell(micro,N0,l,e,D):
    micro_v=np.zeros((N0,N0,N0))
    for i in range(len(micro)):
        print(i)
        c,n,ang=micro[i]
        fill(micro_v,c,n,l,e,1,D)
    return micro_v

def voxelize_ell_n(micro,N0,l,e,D):
    micro_v=np.zeros((N0,N0,N0))
    for i in range(len(micro)):
        print(i)
        c,n,ang=micro[i]
        fill(micro_v,c,n,l,e,i+1,D)
    return micro_v
