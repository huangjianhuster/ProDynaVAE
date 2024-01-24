from sys import argv
# OpenMM Imports
from openmm.unit import *
from openmm.app import *
from openmm import *

# 0) set variables in the simulation
pdb_file = argv[1]
psf_file = argv[2]
out_xml_file = argv[3]
top_inp = '/home2/jianhuang/projects/VAE/ProDynaVAE/forcefield/top_hyres_GPU.inp'
param_inp = '/home2/jianhuang/projects/VAE/ProDynaVAE/forcefield/param_hyres_GPU.inp'

# 1) import coordinates and topology form charmm pdb and psf
print('\nload coordinates, topology and parameters')
pdb = PDBFile(pdb_file)
psf = CharmmPsfFile(psf_file)
top = psf.topology
params = CharmmParameterSet(top_inp, param_inp)
system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)

with open(out_xml_file, 'w') as f:
    f.write(XmlSerializer.serialize(system))
