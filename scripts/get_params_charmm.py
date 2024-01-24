# OpenMM Imports
from openmm.unit import *
from openmm.app import *
from openmm import *

# 0) set variables in the simulation
psf_file = "hyres_example.psf"

# force file
top_inp = '/home2/jianhuang/projects/VAE/ProDynaVAE/forcefield/top_hyres_GPU.inp'
param_inp = '/home2/jianhuang/projects/VAE/ProDynaVAE/forcefield/param_hyres_GPU.inp'

# 1) import coordinates and topology form charmm pdb and psf
print('\nload coordinates, topology and parameters')
psf = CharmmPsfFile(psf_file)
params = CharmmParameterSet(top_inp, param_inp)
system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)

# get internal params
bond_dict = {}
for bond in psf.bond_list:
    bond_dict[(bond.atom1.name, bond.atom2.name)] = (bond.bond_type.req, bond.bond_type.k)
print(bond_dict)

angle_dict = {}
for angle in psf.angle_list:
    angle_dict[(angle.atom1.name, angle.atom2.name, angle.atom3.name)] = (angle.angle_type.theteq, angle.angle_type.k)
print(bond_dict)

dihedral_dict = {}
for dihedral in psf.dihedral_parameter_list:
    dihedral_dict[(dihedral.atom1.name, dihedral.atom1.name, dihedral.atom4.name,  dihedral.atom4.name)] = \
        (dihedral.dihedral_type.phase, dihedral.dihedral_type.per, dihedral.dihedral_type.phi_k)

