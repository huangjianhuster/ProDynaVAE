# Author: Jian Huang
# Date: Oct 9, 2023
# email: jianhuang@umass.edu

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import numpy as np
import matplotlib.pyplot as plt


# get bonds
def get_bonds(psf, xtc, atom1_name, atom2_name, every=1, use_atom_type=True):
    """
    psf: PSF
    xtc: XTC
    atom1_name: atomname of the first atom; ; use atomtype if use_atom_type=True
    atom2_name: atomname of the second atom
    every: frequency of calculation
    use_atom_type: whether use atom type; default=True
    return: bond length array; unit: Angstrom
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_bonds = []
    for ts in u.trajectory[::int(every)]:
        bond_list = []
        for bond in protein_selection.bonds:

            # conditions to select atom names for bonds
            if not use_atom_type:

                atom1 = bond.atoms[0].name
                atom2 = bond.atoms[1].name
                # print(atom1, atom2)
                if atom1 == atom1_name and atom2 == atom2_name:
                    bond_list.append(bond.value())
            else:
                atom1 = bond.atoms[0].type
                atom2 = bond.atoms[1].type
                if atom1 == atom1_name and atom2 == atom2_name:
                    bond_list.append(bond.value())

        all_bonds.append(bond_list)
    return np.array(all_bonds)

# get angles
def get_angles(psf, xtc, atom1_name, atom2_name, atom3_name, every=1, use_atom_type=True):
    """
    psf: PSF
    xtc: XTC
    atom1_name: atomname of the first atom; use atomtype if use_atom_type=True
    atom2_name: atomname of the second atom
    atom3_name: atomname of the third atom
    every: frequency of calculation
    use_atom_type: whether use atom type; default=False
    return: angle list in degrees
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_angles = []
    for ts in u.trajectory[::int(every)]:
        angle_list = []
        for angle in protein_selection.angles:

            # conditions to select atom names for bonds
            if not use_atom_type:
                
                atom1 = angle.atoms[0].name
                atom2 = angle.atoms[1].name
                atom3 =  angle.atoms[2].name
                # print(atom1, atom2)
                if atom1 == atom1_name and atom2 == atom2_name:
                    angle_list.append(angle.value())
            else:
                atom1 = angle.atoms[0].type
                atom2 = angle.atoms[1].type
                atom3 = angle.atoms[2].type
                if atom1 == atom1_name and atom2 == atom2_name:
                    angle_list.append(angle.value())
                
        all_angles.append(angle_list)
    return np.array(all_angles)

# dihedrals
def get_dihedrals(psf, xtc, atom1_name, atom2_name, atom3_name, atom4_name, every=1, use_atom_type=True):
    """
    psf: PSF
    xtc: XTC
    atom1_name: atomname of the first atom; ; use atomtype if use_atom_type=True
    atom2_name: atomname of the second atom
    atom3_name: atomname of the third atom
    atom4_name: atomname of the fourth atom
    every: frequency of querying the trajectory
    use_atom_type: whether use atom type; default=True
    return: dihedral array in degrees
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_dihedrals = []
    for ts in u.trajectory[::int(every)]:
        dihedral_list = []
        for dihedral in protein_selection.dihedrals:

            # conditions to select atom names for bonds
            if not use_atom_type:
                
                atom1 = dihedral.atoms[0].name
                atom2 = dihedral.atoms[1].name
                atom3 =  dihedral.atoms[2].name
                atom4 = dihedral.atoms[3].name
                # print(atom1, atom2)
                if atom1 == atom1_name and atom2 == atom2_name:
                    dihedral_list.append(dihedral.value())
            else:
                atom1 = dihedral.atoms[0].type
                atom2 = dihedral.atoms[1].type
                atom3 = dihedral.atoms[2].type
                atom4 = dihedral.atoms[3].type
                if atom1 == atom1_name and atom2 == atom2_name:
                    dihedral_list.append(dihedral.value())
                
        all_dihedrals.append(dihedral_list)
    return np.array(all_dihedrals)

def get_bb_impropers(psf, xtc, every=1):
    """
    psf: PSF
    xtc: XTC
    every: frequency of calculation
    return: impropers in degrees
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_bbimpropers = []
    for ts in u.trajectory[::int(every)]:
        bbimproper_list = []
        for bbimproper in protein_selection.impropers:         
            atom1 = bbimproper.atoms[0].type
            atom2 = bbimproper.atoms[1].type
            atom3 =  bbimproper.atoms[2].type
            atom4 = bbimproper.atoms[3].type
            # print(atom1, atom2)
            if atom1 == 'C' and atom2 == 'CT1' and atom3 == 'NH1' and atom4 == 'O':
                bbimproper_list.append(bbimproper.value())
            elif atom1 == 'NH1' and atom2 == 'C' and atom3 == 'CT1' and atom4 == 'H':
                bbimproper_list.append(bbimproper.value())
        all_bbimpropers.append(bbimproper_list)
    return np.array(all_bbimpropers)

# get peptide plane omegas
def get_omegas(psf, xtc, every=1):
    """
    psf: PSF
    xtc: XTC
    every: frequency of calculation
    return: array of omega values (in degrees)
    """
    u = mda.Universe(psf, xtc)
    ags = [res.omega_selection() for res in u.residues[:-1]] # ignore the last residues
    R = Dihedral(ags).run(step=int(every))

    return R.results['angles']

# get backbone psi and phi angles
def get_phis_psis(psf, xtc, every=1):
    """
    psf: PSF
    xtc: XTC
    every: frequency of calculation
    return: array of phi values (in degrees), array of psi values (in degrees)
    """
    u = mda.Universe(psf, xtc)
    ags_phi = [res.omega_selection() for res in u.residues[1:]] # the first residue has no phi
    ags_psi = [res.psi_selection() for res in u.residues[:-1]]  # the last residue has no psi

    R_phi = Dihedral(ags_phi).run(step=int(every))
    R_psi = Dihedral(ags_psi).run(step=int(every))
    return R_phi.results['angles'], R_psi.results['angles']

    return R.results['angles']

