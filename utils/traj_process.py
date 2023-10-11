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

#pip install parmed

# only get the protein part of the trajectory
def extract_pro(psf, xtc):
    # determine the output file path
    dirname = os.path.dirname(xtc)
    basename = os.path.basename(xtc)

    # extract
    u = mda.Universe(psf, xtc)
    protein = u.select_atoms('protein')
    protein_psf = protein.convert_to("PARMED")
    out_psf = os.path.join(dirname, basename.split('.')[0] + '_protein.psf')
    out_xtc = os.path.join(dirname, basename.split('.')[0] + '_protein.xtc')
    if os.path.isfile(out_xtc) == False:
        protein_psf.save(out_psf)
        with mda.Writer(out_xtc, protein.n_atoms) as W:
            for ts in u.trajectory:
                W.write(protein)
    # return absolute path
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

def heavy_atom_templete(pdb, new_pdb,new_gro):
    u = mda.Universe(pdb)
    heavy_atoms = u.select_atoms('not name H* and not (resname AMN or resname CBX)')
    heavy_atoms.write(new_pdb)
    heavy_atoms.write(new_gro)
    return None    

def CA_atom_templete(pdb, new_pdb,new_gro):
    u = mda.Universe(pdb)
    CA_atoms = u.select_atoms('name CA')
    CA_atoms.write(new_pdb)
    CA_atoms.write(new_gro)
    return None

# RMSD calculation
def traj_rmsd(psf, xtc, align_select, rmsd_list):
    """
    psf: PSF or TPR
    xtc: already_aligned XTC or DCD
    return rmsd_matrix
        rmsd_matrix has a shape of (4, number_of_frames)
        rmsd_matrix[0]: frames
        rmsd_matrix[1]: time
        rmsd_matrix[2]: rmsd of align_select
        rmsd_matrix[3-]: rmsd of rmsd_list
    """
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    R = mda.analysis.rms.RMSD(u,ref,select=align_select,groupselections=rmsd_list)
    R.run()
    return R.results.rmsd.T

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
    return c_alphas.resnums, rmsf_matrix

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
        Rgyr.append( u.atoms.radius_of_gyration())
    Rgyr = np.array(Rgyr)
    return Rgyr

# helicity & sheet
def traj_ss(psf, xtc):
    if xtc.endswith("xtc"):
        traj = md.load(xtc, top=psf)
    if xtc.endswith("dcd"):
        traj = md.load_dcd(xtc, top=psf)
#    residues = list(traj.topology.residues)
    dssp = md.compute_dssp(traj, simplified=True)
    helicity = np.where(dssp=='H', 1, 0)
    helicity_ave = np.sum(helicity, 0) / helicity.shape[0]
    sheet = np.where(dssp=='E', 1, 0)
    sheet_ave = np.sum(sheet, 0) / sheet.shape[0]
    return  helicity_ave, sheet_ave

# PCA analysis
def traj_pca(psf, xtc):
    pass


# Torison angles
def traj_torsion(psf, xtc):

    return bb_torsion
    pass

# End-to-end distance
def endtoend(psf,dcd):
    u = mda.Universe(psf,dcd)  # always start with a Universe
    nterm = u.select_atoms('protein and name N')[0]  # can access structure via segid (s4AKE) and atom name
    cterm = u.select_atoms('protein and name C')[-1]  # ... takes the last atom named 'C'
    bb = u.select_atoms('protein and backbone')  # a selection (a AtomGroup)
    ete = []
    frame = []
    for ts in u.trajectory:  # iterate through all frames
        r = cterm.position - nterm.position  # end-to-end vector from atom positions
        d = numpy.linalg.norm(r)   # end-to-end distance
        frame.append(ts.frame)
        ete.append(d)
    return frame, ete

# PCA analysis
def PCA(psf,dcd):
    aligner = align.AlignTraj(u, u, select='backbone', in_memory=True).run()
    pc = pca.PCA(u, select='backbone', align=True, mean=None, n_components=None).run()
    backbone = u.select_atoms('backbone')
    n_bb = len(backbone)
    return pc

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






