#!/usr/bin/python
# 
# elast_ref.py
# ANTOINE MARTIN DEC/SESC
# 22 mai 2025
#

from tmfft import *

def compute_ref_elast(micro_name,output_name,K0,G0,K1,G1,N,prec):
	g = Grid(VTKRead(micro_name+".vtk"))
	med=Medium(g,"EPS","SIG")
	med.setElasticBehaviour("IsotropicKG")
	med.declareTemperature("T")
	med.setConstant("T",1000)

	# Déclaration des paramètres
	med.declareParams("K")
	med.declareParams("G")

	# Matrice
	med[0].setConstant("K",K0)
	med[0].setConstant("G",G0)

	# Inclusions
	for i in range(1,N):
		med[i].setConstant("K",K1)
		med[i].setConstant("G",G1)
	setNbOfThreads(6)

	slv=MSolver(med)
	slv.setMaxIterations(500000)
	# Précision du calcul
	slv.setPrecision(prec/g.DivNormFactor())
	slv.setElasticModulusFromKG((K0+K1)/2,(G0+G1)/2)

	VTKout=slv.setVTKout()
	VTKout.setBasename(output_name)
	VTKout.matID()
	VTKout.newField("SIG")
	VTKout.newField("EPS")

	slv.tangentMatrix(1,1.,output_name)
	
