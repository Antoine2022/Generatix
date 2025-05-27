#!/usr/bin/python
# 
# EP_tmfft.py
# ANTOINE MARTIN DEC/SESC
# 23 mai 2025
#

import numpy as np
import time
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as vnp
from tmfft import *

def read_vtk(file_name, champ_name):
    reader = vtkStructuredPointsReader()
    reader.SetFileName(file_name)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    vec = [i - 1 for i in dim]
    # print(tuple(vec[:]))
    e0 = vnp.vtk_to_numpy(data.GetCellData().GetArray(champ_name))
    return e0.reshape(vec, order="F")


Id=np.eye(6)
J=np.zeros((6,6))
for i in range(3):
    for j in range(3):
        J[i,j]=1/3
K=Id-J

def compute_mode_0(micro_name,Ni,K0,G0):
    g = Grid(VTKRead(micro_name+".vtk"))
    med=Medium(g,"EPS","SIG")
    med.setElasticBehaviour("IsotropicKG")
    med.declareTemperature("T")
    med.setConstant("T",1000)
    med.declareParams("K")
    med.declareParams("G")
    med[0].setConstant("K",K0)
    med[0].setConstant("G",G0)
    for j,E in enumerate(["EXX","EYY","EZZ","GXY","GXZ","GYZ"]):
        for i in range(1,Ni+1):
            med[i].setConstant("K",K0/3+K0)
            med[i].setConstant("G",0.5*K0+G0)
        setNbOfThreads(5)
        slv=MSolver(med)
        slv.setPrecision(1e-6/g.DivNormFactor())
        slv.setElasticModulusFromKG(K0,G0)
        slv.setMaxIterations(1)
        VTKout = slv.setVTKout()
        VTKout.setBasename("modes/tmp_it="+str(1)+"_"+E)
        VTKout.newField("EPS")
        VTKout.matID()
        E_=np.zeros((6,))
        E_[j]=1
        print(E_)
        slv.setEvolution("EXX",{0:E_[0]})
        slv.setEvolution("EYY",{0:E_[1]})
        slv.setEvolution("EZZ",{0:E_[2]})
        slv.setEvolution("EXY",{0:E_[3]})
        slv.setEvolution("EXZ",{0:E_[4]})
        slv.setEvolution("EYZ",{0:E_[5]})
        output="tmp.res"
        slv.loading([0],0,output)
        

def compute_modes(micro_name,Ni,nb,K0,G0,K1,G1):
    g = Grid(VTKRead(micro_name+".vtk"))
    med=Medium(g,"EPS","SIG")
    med.setElasticBehaviour("IsotropicKG")
    med.declareTemperature("T")
    med.setConstant("T",1000)
    med.declareParams("K")
    med.declareParams("G")
    med[0].setConstant("K",K0)
    med[0].setConstant("G",G0)
    for i in range(1,Ni+1):
        med[i].setConstant("K",K1)
        med[i].setConstant("G",G1)
    for j,E in enumerate(["EXX","EYY","EZZ","GXY","GXZ","GYZ"]):
        file_loc=VTKRead("./modes/tmp_it="+str(1)+"_"+E+"_01.vtk")
        med.FromVTK(file_loc,"EPS")
        setNbOfThreads(5)
        slv=MSolver(med)
        slv.setPrecision(1e-6/g.DivNormFactor())
        slv.setElasticModulusFromKG(K0,G0)
        slv.setMaxIterations(nb-1)
        VTKout = slv.setVTKout()
        VTKout.setBasename("modes/tmp_it="+str(2)+"_"+E)
        VTKout.newField("EPS")
        VTKout.matID()
        E_=np.zeros((6,))
        E_[j]=1
        print(E_)
        slv.setEvolution("EXX",{0:E_[0]})
        slv.setEvolution("EYY",{0:E_[1]})
        slv.setEvolution("EZZ",{0:E_[2]})
        slv.setEvolution("EXY",{0:E_[3]})
        slv.setEvolution("EXZ",{0:E_[4]})
        slv.setEvolution("EYZ",{0:E_[5]})
        output="tmp.res"
        slv.loading([0],0,output)


def compute_min_EP(micro_name,N0,nb,Eb,K0,G0,K1,G1):
    dC=3*(K1-K0)*J+2*(G1-G0)*K
    C0=3*K0*J+2*G0*K
    
    N = N0 * np.ones(3, dtype=np.int64)
    khi = read_vtk(micro_name+".vtk", "MaterialId")
    frac = 0.0
    for i in np.ndindex(tuple(N)):
        if khi[i] == 1:
            frac += 1 / N0**3
    print(frac)
    print("reading...")
    A = np.zeros((nb,) + tuple(N) + (6, 6))
    for k in range(nb):
        for j, E in enumerate(["EXX","EYY","EZZ","GXY","GXZ","GYZ"]):
            if k==0:
                champ = "./modes/tmp_it=" + str(1) + "_" + E + "_01.vtk"
            else:
                champ = "./modes/tmp_it=" + str(2) + "_" + E + "_0"+str(k)+".vtk"
            for i, eps in enumerate(["EXX","EYY","EZZ","GXY","GXZ","GYZ"]):
                if i==3 or i==4 or i==5:
                    A[k, :, :, :, i, j] = -0.5*read_vtk(champ, eps)
                else:
                    A[k, :, :, :, i, j] = -read_vtk(champ, eps)
    
    t0 = time.time()
    d=6
    print("compute matrice UB")
    list_H = [np.zeros((d,)) for _ in range(nb)]
    list_G = [[np.zeros((d, d)) for _ in range(nb)] for __ in range(nb)]
    for i in np.ndindex(tuple(N)):
        C=C0+khi[i]*dC
        for k in range(nb):
            if i==(0,0,0):
                print(k)
            list_H[k] += np.dot((A[k][i]).transpose(),np.dot(C,Eb)) / N0**3
            for l in range(k, nb):
                list_G[k][l] += np.dot((A[k][i]).transpose(),np.dot(C,A[l][i])) / N0**3
    print('voilà',list_H[0][0],list_H[1][0],list_G[0][0][0,0],list_G[1][1][0,0])
    for k in range(nb):
        for l in range(k):
            list_G[k][l] = np.transpose(list_G[l][k])

    print("compute UB")
    MAT = np.zeros((nb * d, nb * d))
    U = np.zeros((nb * d,))
    for k in range(nb):
        U[k * d : (k + 1) * d] = list_H[k]  # np.dot(list_H[k],Eb)
        for l in range(nb):
            MAT[k * d : (k + 1) * d, l * d : (l + 1) * d] = list_G[k][l]

    CV=C0+frac*dC
    sig_UB = np.zeros((nb,))
    print("time", time.time() - t0)
    print(CV[0,0])
    for k in range(nb):
        tau_ = np.dot(np.linalg.inv(MAT[: (k + 1) * d, : (k + 1) * d]), U[: (k + 1) * d])
        sig_UB[k] = np.dot(Eb,np.dot(CV,Eb)) - np.dot(tau_[: (k + 1) * d], U[: (k + 1) * d])
        print(sig_UB[k])

    # tau = np.zeros(tuple(N) + (d, d))
    # for i in np.ndindex(tuple(N)):
    #     tau[i] = khi[i] * np.eye(d)
    # T_ = np.zeros((nb,) + tuple(N) + (d, d))
    # T_[0] = tau
    # k = 0
    # while k < nb - 1:
    #     print("compute T" + str(k + 1))
    #     # H[k]=Gamma(T,N,k0)
    #     for i in np.ndindex(tuple(N)):
    #         T_[k + 1][i] = -khi[i] * A[k][i]
    #     k += 1

    # F = np.zeros((nb,) + tuple(N) + (d, d))
    # for k in range(nb):
    #     print("compute F" + str(k))
    #     T_m = 0.0
    #     for i in np.ndindex(tuple(N)):
    #         T_m += T_[k][i] / N0**d
    #     for i in np.ndindex(tuple(N)):
    #         F[k][i] = -T_[k][i] + T_m * np.eye(d) + A[k][i]

    # print("compute LB...")
    # list_G = [[np.zeros((d, d)) for _ in range(nb)] for __ in range(nb)]
    # list_H = [np.zeros((d, d)) for _ in range(nb)]
    # rr = 0.0
    # for i in np.ndindex(tuple(N)):
    #     rho = 1 / (1 + l0 * khi[i])
    #     for k in range(nb):
    #         list_H[k] += rho * (F[k][i]).transpose() / N0**d
    #         for l in range(k, nb):
    #             list_G[k][l] += rho * np.dot((F[k][i]).transpose(), F[l][i]) / N0**d
    #     rr += rho / N0**d
    # for k in range(nb):
    #     for l in range(k):
    #         list_G[k][l] = np.transpose(list_G[l][k])

    # MAT = np.zeros((nb * d, nb * d))
    # U = np.zeros((nb * d, d))
    # for k in range(nb):
    #     U[k * d : (k + 1) * d] = list_H[k]  # np.dot(list_H[k],Jb)
    #     for l in range(nb):
    #         MAT[k * d : (k + 1) * d, l * d : (l + 1) * d] = list_G[k][l]

    # sig_LB = np.zeros((nb, d, d))
    # print("time", time.time() - t0)
    # for k in range(nb):
    #     tau_ = np.dot(
    #         np.linalg.inv(MAT[: (k + 1) * d, : (k + 1) * d]), U[: (k + 1) * d]
    #     )
    #     sig_LB[k] = np.linalg.inv(
    #         rr * np.eye(d) - np.dot((tau_[: (k + 1) * d]).transpose(), U[: (k + 1) * d])
    #     )
    #     print(sig_LB[k])

N0=128
nb=5
Eb=np.array([1.,0.,0.,0.,0.,0.])
K0=1.
G0=1.
K1=100.
G1=100.
micro_name="micro_hard_sph_exc=0.01_ratio=0.06_frac=0.3"
compute_mode_0("./microstructures/"+micro_name,1,K0,G0)
compute_modes("./microstructures/"+micro_name,1,nb,K0,G0,K1,G1)
compute_min_EP("./microstructures/"+micro_name,N0,nb,Eb,K0,G0,K1,G1)