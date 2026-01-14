import numpy as np
from spherocylinders import *
    
l=0.5 #length of cylinder
e=10 #aspect ratio
D=1 #side of the box
f=0.12 # target fraction (will be smaller after voxellization)

micro=generate_micro(D/2,l,l/e/2,f)
micro_name="micro_randomly_oriented"
np.save("./"+micro_name+".npy",micro)

from voxel import *
size=128
micro_v=voxelize(micro,size,l,e,D,"cylinder")
write_vtk(micro_v,micro_name,size)

f=0.1 # target fraction (will be smaller after voxellization)
micro=generate_micro_aligned(D/2,l,l/e/2,f)
micro_name="micro_aligned"
np.save("./"+micro_name+".npy",micro)
micro_v=voxelize(micro,size,l,e,D,"spheroid")
write_vtk(micro_v,micro_name,size)