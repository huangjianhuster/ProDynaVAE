# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
import MDAnalysis.analysis.rms as rms
import mdtraj as md
import numpy as np
import os

# only get the protein part of the trajectory
def extract_pro(psf, xtc):
    # determine the directory path
    # os.path.dirname()
    u = mda.Universe(psf, xtc)
    protein = u.select_atoms('protein')
    protein_psf = protein.convert_to("PARMED")
    out_psf = psf.split('.')[0] + '_protein.psf'
    out_xtc = xtc.split('.')[0] + '_protein.xtc'
    protein_psf.save(out_psf)

    with mda.Writer(out_xtc, protein.n_atoms) as W:
        for ts in u.trajectory:
            W.write(protein)

    return out_psf, out_xtc

# protein centering and alignment
def traj_align(psf, xtc, out_xtc, center=True):
    """
    psf: PSF; TPR for the simulation system;
    xtc: XTC or DCD format;
    out_xtc: aligned output trajectory path;
    center: whether center the protein to the center of the box;

    !!! default: align the trajectory with respect to the first frame [hard coded!]
    return None
    """ 
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    
    # Center protein in the center of the box
    if center:
        protein = u.select_atoms('protein')
        not_protein = u.select_atoms('not protein')
        for ts in u.trajectory:
            protein.unwrap(compound='fragments')
            protein_center = protein.center_of_mass(pbc=True)
            dim = ts.triclinic_dimensions
            box_center = np.sum(dim, axis=0) / 2
            # translate all atoms
            u.atoms.translate(box_center - protein_center)
            # wrap all solvent part back to the box
            not_protein.wrap(compound='residues')

    # align using C-alpha atoms
    align.AlignTraj(u, # universe object; trajectory to align
                    ref, # reference
                    select='name CA', # selection of atoms to align
                    filename=out_xtc,
                    match_atoms=True,
                   ).run()

    return None

def traj_align_onfly(psf, xtc, out_xtc, center=True):
    """
    psf: PSF; TPR for the simulation system;
    xtc: XTC or DCD format;
    out_xtc: aligned output trajectory path;
    center: whether center the protein to the center of the box;

    !!! default: align the trajectory with respect to the first frame [hard coded!]
    return None
    """
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    if center:
        protein = u.select_atoms('protein')
        not_protein = u.select_atoms('not protein')
        transforms = [trans.unwrap(protein),
                trans.center_in_box(protein, wrap=True),
                trans.wrap(not_protein)]

        u.trajectory.add_transformations(*transforms)
    
    # align using C-alpha atoms
    align.AlignTraj(u, # universe object; trajectory to align
                    ref, # reference
                    select='name CA', # selection of atoms to align
                    filename=out_xtc,
                    match_atoms=True,
                   ).run()

    return None

# RMSD calculation
def traj_rmsd(psf, xtc):
    """
    psf: PSF or TPR
    xtc: already_aligned XTC or DCD
    return rmsd_matrix
        rmsd_matrix has a shape of (4, number_of_frames)
        rmsd_matrix[0]: frames
        rmsd_matrix[1]: time
        rmsd_matrix[2]: rmsd of C-alpha
        rmsd_matrix[4]: rmsd of protein
    """
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    R = rms.RMSD(u, ref, select="name CA", groupselections=["protein",])
    R.run()
    rmsd_matrix = R.rmsd.T
    return rmsd_matrix

# RMSF calculation: only for C-alpha
def traj_rmsf(psf, xtc):
    """
    psf: PSF or TPR
    xtc: already_aligned XTC or DCD
    return resid, rmsf_matrix
    """
    u = mda.Universe(psf, xtc)
    # ref = mda.Universe(psf, xtc)
    # ref.trajectory[0]
    c_alphas = u.select_atoms('protein and name CA')
    R = rms.RMSF(c_alphas).run()
    rmsf_matrix = R.results.rmsf
    return c_alphas.resids, rmsf_matrix

# radius of gyration (IDP protein)
def traj_rg(psf, xtc):
    """
    psf: PSF or TPR
    xtc: XTC or DCD
    return Rgyr
    """
    u = mda.Universe(psf, xtc)
    Rgyr = []
    for ts in u.trajectory:
        Rgyr.append(u.trajectory.time, u.atoms.radius_of_gyration())
    Rgyr = np.array(Rgyr)
    return Rgyr

# helicity & sheet
def traj_ss(psf, xtc):
    if xtc.endswith("xtc"):
        traj = md.load(xtc, top=psf)
    if xtc.endswith("dcd"):
        traj = md.load_dcd(xtc, top=psf)
    residues = list(traj.topology.residues)
    dssp = md.compute_dssp(traj, simplified=True)
    helicity = np.where(dssp=='H', 1, 0)
    helicity_ave = np.sum(helicity, 0) / helicity.shape[0]
    sheet = np.where(dssp=='E', 1, 0)
    sheet_ave = np.sum(sheet, 0) / sheet.shape[0]
    return residues, helicity_ave, sheet_ave

# PCA analysis
def traj_pca(psf, xtc):
    pass



# write pdbs into trajectory <-- for VAE decoder-generated trajectory
def pdbs2xtc(pdblist, out_xtc):
    """
    pdblist: list of pdb paths to be combined together into a single trajectory file
    out_xtc: output trajectory path
    """
    # to determin PDB atoms
    u = mda.Universe(pdblist[0])
    with mda.Writer(out_xtc, len(u.atoms)) as xtc_writer:
        for pdb in pdblist:
            u.load_new(pdb)
            xtc_writer.write(u)

    return None






