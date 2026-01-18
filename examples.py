import numpy as np
from spherocylinders import *
    
l=0.5 #length of cylinder
e=10 #aspect ratio
D=1 #side of the box
f=0.03 # target fraction (will be smaller after voxellization)

micro=generate_micro(D/2,np.array([[l,l/e/2,f]]),"random")
micro_name="micro_randomly_oriented"
np.save("./"+micro_name+".npy",micro)
micro_poly=generate_micro(D/2,np.array([[l,l/e/2,f],[l/2,l/e/4,f/3]]),"random")
micro_name="micro_poly_randomly_oriented"
np.save("./"+micro_name+".npy",micro_poly)

from voxel import *
size=128
micro_v=voxelize(micro_poly,size,D,"cylinder")
write_vtk(micro_v,micro_name+"_cylinders",size)
micro_v=voxelize(micro,size,D,"spherocylinder")
write_vtk(micro_v,micro_name+"_spherocylinders",size)

f=0.1 # target fraction (will be smaller after voxellization)
micro=generate_micro(D/2,np.array([[l,l/e/2,f]]),"aligned")
micro_name="micro_aligned"
np.save("./"+micro_name+".npy",micro)
micro_v=voxelize(micro,size,D,"spheroid")
write_vtk(micro_v,micro_name,size)