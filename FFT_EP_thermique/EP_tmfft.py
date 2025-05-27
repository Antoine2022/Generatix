import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import vtk
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as vnp


path = "/home/am280701/codes/TMFFT_dev/examples/FFT_Seb/"
d = 3
nb = 4
N0 = 128
ratio = 0.06


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


N = N0 * np.ones(d, dtype=np.int64)


for frac0 in [
    0.781,
    0.86,
    0.93,
    0.963,
]:  # [0.047,0.255,0.324,0.357,0.42,0.469,0.55,0.705,0.781,0.86,0.93,0.963]:
    name = "micro_spheres_frac=" + str(frac0)
    # name="micro_hard_sph_exc=0.01_ratio=0.06_frac=0.3"
    micro = path + "champ_" + name + "_it=1_GX_00.vtk"
    khi = read_vtk(micro, "lam")
    frac = 0.0
    for i in np.ndindex(tuple(N)):
        khi[i] = khi[i] - 1
        if khi[i] == 1:
            frac += 1 / N0**d
    print(frac)

    A = np.zeros((nb,) + tuple(N) + (d, d))
    for k in range(nb):
        for j, G in enumerate(["GX", "GY", "GZ"]):
            champ = path + "champ_" + name + "_it=" + str(k + 1) + "_" + G + "_00.vtk"
            for i in range(d):
                # A_k_ij=read_vtk(champ,'grdT['+str(j)+']')
                A[k, :, :, :, i, j] = -read_vtk(champ, "grdT[" + str(i) + "]")

    # Eb=np.array([1.,0.,0.])
    c_i = 1e2
    l0 = c_i - 1
    t0 = time.time()

    print("compute matrice UB")
    list_G = [[np.zeros((d, d)) for _ in range(nb)] for __ in range(nb)]
    list_H = [np.zeros((d, d)) for _ in range(nb)]
    cv = 0.0
    for i in np.ndindex(tuple(N)):
        sig = 1 + l0 * khi[i]
        for k in range(nb):
            list_H[k] += sig * (A[k][i]).transpose() / N0**d
            for l in range(k, nb):
                list_G[k][l] += sig * np.dot((A[k][i]).transpose(), A[l][i]) / N0**d
        cv += sig / N0**d
    # print('voilà',list_H[0][0,0],list_H[1][0,0],list_G[0][0][0,0],list_G[1][1][0,0])

    for k in range(nb):
        for l in range(k):
            list_G[k][l] = np.transpose(list_G[l][k])

    print("compute UB")
    MAT = np.zeros((nb * d, nb * d))
    U = np.zeros((nb * d, d))
    for k in range(nb):
        U[k * d : (k + 1) * d] = list_H[k]  # np.dot(list_H[k],Eb)
        for l in range(nb):
            MAT[k * d : (k + 1) * d, l * d : (l + 1) * d] = list_G[k][l]

    sig_UB = np.zeros((nb, d, d))
    print("time", time.time() - t0)
    for k in range(nb):
        tau_ = np.dot(
            np.linalg.inv(MAT[: (k + 1) * d, : (k + 1) * d]), U[: (k + 1) * d]
        )
        sig_UB[k] = cv * np.eye(d) - np.dot(
            (tau_[: (k + 1) * d]).transpose(), U[: (k + 1) * d]
        )
        print(sig_UB[k])
    sigma_UB = sig_UB[nb - 1]

    # Jb=np.array([1.,0.,0.])

    tau = np.zeros(tuple(N) + (d, d))
    for i in np.ndindex(tuple(N)):
        tau[i] = khi[i] * np.eye(d)
    T_ = np.zeros((nb,) + tuple(N) + (d, d))
    T_[0] = tau
    k = 0
    while k < nb - 1:
        print("compute T" + str(k + 1))
        # H[k]=Gamma(T,N,k0)
        for i in np.ndindex(tuple(N)):
            T_[k + 1][i] = -khi[i] * A[k][i]
        k += 1

    F = np.zeros((nb,) + tuple(N) + (d, d))
    for k in range(nb):
        print("compute F" + str(k))
        T_m = 0.0
        for i in np.ndindex(tuple(N)):
            T_m += T_[k][i] / N0**d
        for i in np.ndindex(tuple(N)):
            F[k][i] = -T_[k][i] + T_m * np.eye(d) + A[k][i]

    print("compute LB...")
    list_G = [[np.zeros((d, d)) for _ in range(nb)] for __ in range(nb)]
    list_H = [np.zeros((d, d)) for _ in range(nb)]
    rr = 0.0
    for i in np.ndindex(tuple(N)):
        rho = 1 / (1 + l0 * khi[i])
        for k in range(nb):
            list_H[k] += rho * (F[k][i]).transpose() / N0**d
            for l in range(k, nb):
                list_G[k][l] += rho * np.dot((F[k][i]).transpose(), F[l][i]) / N0**d
        rr += rho / N0**d
    for k in range(nb):
        for l in range(k):
            list_G[k][l] = np.transpose(list_G[l][k])

    MAT = np.zeros((nb * d, nb * d))
    U = np.zeros((nb * d, d))
    for k in range(nb):
        U[k * d : (k + 1) * d] = list_H[k]  # np.dot(list_H[k],Jb)
        for l in range(nb):
            MAT[k * d : (k + 1) * d, l * d : (l + 1) * d] = list_G[k][l]

    sig_LB = np.zeros((nb, d, d))
    print("time", time.time() - t0)
    for k in range(nb):
        tau_ = np.dot(
            np.linalg.inv(MAT[: (k + 1) * d, : (k + 1) * d]), U[: (k + 1) * d]
        )
        sig_LB[k] = np.linalg.inv(
            rr * np.eye(d) - np.dot((tau_[: (k + 1) * d]).transpose(), U[: (k + 1) * d])
        )
        print(sig_LB[k])
    sigma_LB = sig_LB[nb - 1]

    for j, CC in enumerate(["XX", "YY", "ZZ"]):
        file = open(
            "./data/tmfft_spheres_penetrables_c="
            + str(c_i)
            + "_N0="
            + str(N0)
            + "_ratio="
            + str(ratio)
            + "_"
            + CC
            + ".txt",
            "a",
        )
        # file=open('./data/spheres_hard_c='+str(c_i)+'_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
        file.write(str(frac))  # +' '+str(sigma_HS_LB)+' '+str(sigma_HS_UB))
        for k in range(nb):
            file.write(" " + str(sig_LB[k][j, j]) + " " + str(sig_UB[k][j, j]))
        file.write("\n")
        file.flush()
        # file=open('../data/c1000/ellipsoides_orientees_HS0_aspect='+str(aspect)+'_penetrables_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
        # file.write(str(frac)+' '+str(HS0)+'\n')
        # file=open('../data/FFT/c1000/reference_ellipsoides_orientees_aspect='+str(aspect)+'_penetrables_N0='+str(N0)+'_ratio='+str(ratio)+'.txt','a')
        # file.write(str(frac)+' '+str(ref)+'\n')
