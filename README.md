# Generatix

This repository contains some tools for generation of microstructures.

spherocylinders.py : tools for generation of a microstructure of hard spherocylinders, with RSA (requires numba)

voxel.py : tools for voxelization of a microstructure (requires numba, and vtk is used for visualization but can be removed)

examples.py : demos

## Examples

With spherocylinders.py, the following microstructures were generated (the visualization was obtained with a construction via Comsol Multiphysics[^1]):
 
<p align="center">
  <img src="images/cellule_dense.PNG" width="60%" /><br>
  <em>Conditions périodiques, hard cylinders</em>
</p>

<p align="center">
  <img src="images/cellule_orient.PNG" width="60%" /><br>
  <em>Conditions périodiques, hard cylinders, orientation unique</em>
</p>

<p align="center">
  <img src="images/sphere_dense.jpg" width="60%" /><br>
  <em>Hard cylinders dans une sphère</em>
</p>


[^1]: COMSOL Multiphysics® v. 5.6. www.comsol.com. COMSOL AB, Stockholm, Sweden