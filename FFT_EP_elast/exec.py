#!/usr/bin/python
# 
# visco_ref.py
# ANTOINE MARTIN DEC/SESC
# 23 mai 2025
#
from microstructure_periodique import *
from voxelize import *
from elast_ref import *


K0=1.
G0=1.
K1=100.
G1=100.

f=0.05
e=20.
l=0.5
D=1.

def write_vtk(mic,name,N):
	VTK_HEADER = f"""# vtk DataFile Version 4.5
Materiau
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS    {N+1}   {N+1}   {N+1}
ORIGIN    0.000   0.000   0.000
SPACING   {1.:.7e} {1.:.7e} {1.:.7e}
CELL_DATA   {N*N*N}
SCALARS MaterialId unsigned_short
LOOKUP_TABLE default
"""
	data = np.zeros((N, N, N), dtype=">u2")
	for i in np.ndindex((N,N,N)):
		ii,jj,kk=i
		data[ii,jj,kk]=mic[kk,jj,ii]
	with open(name+".vtk", "wb") as file:
		file.write(VTK_HEADER.encode())
		data.tofile(file)


ratio=l/D
siz=128
prec=1e-3
micro_name="micro_hard_sph_exc=0.01_ratio=0.06_frac=0.3"
#micro_name="micro_hard_ell_ratio="+str(ratio)+"_e="+str(e)+"_frac="+str(f)
#micro=genere_micro(D/2,l,l/e/2,f)
#np.save("./microstructures/"+micro_name+".npy",micro)
#params=np.array([K0,G0,K1,G1,f,e,l,D,siz,prec])
#np.save("./params/param_"+micro_name+".npy",params)
#micro=np.load("./microstructures/"+micro_name+".npy")
#micro_v=voxelize_ell_n(micro,siz,l,e,D)
#Ni=len(micro)
Ni=1
#print(micro_v.shape)
#write_vtk(micro_v,"./microstructures/"+micro_name,siz)

#output_name="ref_ell_ratio="+str(ratio)+"_e="+str(e)+"_frac="+str(f)
output_name='tmp'
compute_ref_elast("./microstructures/"+micro_name,"./fields/"+output_name,K0,G0,K1,G1,Ni+1,prec)
