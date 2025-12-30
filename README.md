# Generatix

This repository contains some tools for generation of microstructures.

spherocylinders.py : tools for generation of a microstructure of hard spherocylinders, with RSA (requires numba)

voxel.py : tools for voxelization of a microstructure (requires numba, and vtk is used for visualization but can be removed)

examples.py : demos

## Examples

With spherocylinders.py, the following microstructures were generated (the visualization was obtained with a construction via Comsol Multiphysics[^1]):

![](cellule_dense.png)
![](cellule_orient.png)
![](sphere_dense.jpg)

[^1]: COMSOL Multiphysics® v. 5.6. www.comsol.com. COMSOL AB, Stockholm, Sweden