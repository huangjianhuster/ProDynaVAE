# Author: Jian Huang
# Date: 2023-11-30
# E-mail: jianhuang@umass.edu

# TODO:
# 1. class Ensemble for general analysis: RMSD, RMSF etc
# 2. class IDPEnsemble for IDPs
# 3. class FoldEnsemble for folded proteins

# Dependencies
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import warnings
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis import rms, pca
import mdtraj as md
from multiprocessing import Pool
from functools import partial
import os
import psutil
import subprocess
from Bio.PDB import PDBParser, DSSP
import parmed as pmd

mda.warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning) 

three2one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'HSD': 'H'}

class Ensemble:
    def __init__(self, psf_file, xtc_file, top_file=None):
        self.psf = psf_file
        self.xtc = xtc_file
        if top_file:
            self.load_top(top_file)
        self.universe = mda.Universe(psf_file, xtc_file)
        self.ref_universe = mda.Universe(psf_file, xtc_file)
        self.protein = self.universe.select_atoms("protein")
        self.resid = self.get_resid()
        self.sequence = self.get_sequence()
        self.n_frames = self.universe.trajectory.n_frames
        self.dt = self.universe.trajectory.dt


    def get_resid(self):
        return [res.resid for res in self.universe.residues]
    
    def get_sequence(self):
        return "".join([three2one[res.resname] for res in self.universe.residues])

    def filter_atoms(self, atom_selection):
        """Filter atoms based on a selection string."""
        self.atoms = self.universe.select_atoms(atom_selection)

    def get_rmsd(self, align_select, rmsd_list):
        """
        align_select: (str), selection syntax in mdanalysis for the alignment
        rmsd_list: (list), selections for different parts 
        return: rmsd_matrix
            rmsd_matrix has a shape of (3+len(rmsd_list), number_of_frames)
            rmsd_matrix[0]: frames
            rmsd_matrix[1]: time
            rmsd_matrix[2]: rmsd of align_select
            rmsd_matrix[3-]: rmsd of rmsd_list
        """
        R = rms.RMSD(self.universe, self.ref_universe, select=align_select, groupselections=rmsd_list)
        R.run()
        return R.results.rmsd.T
    
    def get_rmsf(self):
        """
        RMSF calculaton (default: C-alpha atoms)
        return residue_index_array, rmsf_array
        """
        c_alphas = self.universe.select_atoms('protein and name CA')
        R = rms.RMSF(c_alphas).run()
        rmsf_array = R.results.rmsf
        return c_alphas.resnums, rmsf_array
        
    def get_rg(self, n_threads=None):
        """
        return Rgyr # [[RG_whole, RG_x, RG_y, RG_z]]
        """
        run_per_frame = partial(radgyr_per_frame,
                                atomgroup=self.protein,
                                masses=self.protein.masses)
        
        if n_threads:
            self.available_threads = self.get_available_threads()
        else:
            self.available_threads = n_threads
        
        with Pool(self.available_threads) as worker_pool:
            result = worker_pool.map(run_per_frame, np.arange(self.n_frames))
        rg = np.asarray(result).T
        self.rg = rg
        return rg

    @staticmethod
    def get_available_threads():
        """Get the number of available threads in the current environment."""
        total_threads = os.cpu_count()
        used_threads = psutil.cpu_count(logical=False)  # Physical cores
        available_threads = total_threads - used_threads
        return available_threads
    
    def get_ss_deprecated(self):
        ss = []
        for i in np.arange(self.n_frames):
            res_idx, dssp = dssp_per_frame(i, self.protein)
            ss.append(dssp)
        return res_idx, dssp
    
    def get_ss(self):
        if self.xtc.endswith("xtc"):
            traj = md.load(self.xtc, top=self.psf)
        if self.xtc.endswith("dcd"):
            traj = md.load_dcd(self.xtc, top=self.psf)
        dssp = md.compute_dssp(traj, simplified=True)
        helicity = np.where(dssp=='H', 1, 0)
        helicity_ave = np.sum(helicity, 0) / helicity.shape[0]
        sheet = np.where(dssp=='E', 1, 0)
        sheet_ave = np.sum(sheet, 0) / sheet.shape[0]
        return  {"helix": helicity_ave, "sheet": sheet_ave}
    
    def get_end2end(self, n_threads=None):
        run_per_frame = partial(end2end_per_frame,
                                atomgroup=self.protein)
        if n_threads:
            self.available_threads = self.get_available_threads()
        else:
            self.available_threads = n_threads
        
        with Pool(self.available_threads) as worker_pool:
            result = worker_pool.map(run_per_frame, np.arange(self.n_frames))
        end2end = np.asarray(result)
        return end2end
    
    def pca(self, selection="protein and backbone", n_components=None):
        """
        selection: selection syntax str in MDAnalysis
        n_compnents: int; project coordinates in a reduced dimension
        return: PCA results # a dict with "variance", "cumulated_variance" and "p_components"

        if n_components is given, it will give the transformed points in the reduced dimensions

        # ref: https://userguide.mdanalysis.org/stable/examples/analysis/reduced_dimensions/pca.html
        (using "selection" as the alignment and also output only the "selection" atoms)
        """
        pc = pca.PCA(self.universe, select=selection, align=True, mean=None, n_components=None).run()
        if n_components:
            return pc.results
        else:
            return pc.transform(self.universe.select_atoms(selection), n_components=n_components)
        
    # covalent geometry
    def load_top(self, top_file):
        self.top = pmd.load_file(top_file)
        self.bond_types = self.top.parameterset.bond_types
        self.angle_types = self.top.parameterset.angle_types
        self.dihedral_types = self.top.parameterset.dihedral_types
        self.cmap_types = self.top.parameterset.cmap_types
        self.improper_types = self.top.parameterset.improper_types
        return None
    
    def get_bond_eq(self, atom1_type, atom2_type):
        r_eq = self.bond_types[(atom1_type, atom2_type)]
        return r_eq

    def get_angle_eq(self, atom1_type, atom2_type, atom3_type):
        r_eq = self.angle_types[(atom1_type, atom2_type, atom3_type)]
        return r_eq

    def get_dihedral_eq(self, atom1_type, atom2_type, atom3_type, atom4_type):
        r_eq = self.dihedral_types[(atom1_type, atom2_type, atom3_type, atom4_type)]
        return r_eq
    
    def get_improper_eq(self, atom1_type, atom2_type, atom3_type, atom4_type):
        r_eq = self.improper_types[(atom1_type, atom2_type, atom3_type, atom4_type)]
        return r_eq
        
    def get_bonds(self, selection, atom1_name, atom2_name, n_threads=None):
        """
        selection: MDAnalysis selection syntax; example "protein and resid 10"
        atom1_type: atomname of the first atom;
        atom2_type: atomname of the second atom;
        n_threads: number of CPU threads;
        return: bond length array # unit: Angstrom
        """
        atom_group = self.universe.select_atoms(selection)
        run_per_frame = partial(bonds_per_frame,
                        atomgroup = atom_group, 
                        atom1_name = atom1_name, 
                        atom2_name = atom2_name)
        
        if n_threads:
            self.available_threads = self.get_available_threads()
        else:
            self.available_threads = n_threads
        
        with Pool(self.available_threads) as worker_pool:
            result = worker_pool.map(run_per_frame, np.arange(self.n_frames))
        bonds = np.asarray(result)
        return bonds
    
    def get_angles(self, selection, atom1_name, atom2_name, atom3_name, n_threads=None):
        """
        selection: MDAnalysis selection syntax; example "protein and resid 10"
        atom1_type: atom type of the first atom;
        atom2_type: atom type of the second atom;
        atom3_type: atom type of the third atom;
        n_threads: number of CPU threads;
        return: angle array # unit: degree
        """
        atom_group = self.universe.select_atoms(selection)
        run_per_frame = partial(angles_per_frame,
                        atomgroup=atom_group,
                        atom1_name = atom1_name, 
                        atom2_name = atom2_name,
                        atom3_name = atom3_name)
        
        if n_threads:
            self.available_threads = self.get_available_threads()
        else:
            self.available_threads = n_threads
        
        with Pool(self.available_threads) as worker_pool:
            result = worker_pool.map(run_per_frame, np.arange(self.n_frames))
        angles = np.asarray(result)
        return angles

    def get_dihedrals(self, atom1_type, atom2_type, atom3_type, atom4_type, n_threads=None):
        """
        get a certain dihedral type (may includes many dihedrals from different residues) from the whole protein

        atom1_type: atomname of the first atom;
        atom2_type: atomname of the second atom;
        atom3_type: atomname of the third atom;
        atom4_type: atomname of the fourth atom;
        n_threads: number of CPU threads;
        return: dihedral array # unit: degree
        """
        run_per_frame = partial(dihedrals_per_frame,
                        atomgroup=self.protein,
                        atom1_type=atom1_type, 
                        atom2_type=atom2_type,
                        atom3_type=atom3_type,
                        atom4_type=atom4_type,)
        
        if n_threads:
            self.available_threads = self.get_available_threads()
        else:
            self.available_threads = n_threads
        
        with Pool(self.available_threads) as worker_pool:
            result = worker_pool.map(run_per_frame, np.arange(self.n_frames))
        dihedrals = np.asarray(result)
        return dihedrals

    def get_atoms_info(self, resid):
        """
            for a give resid, get the atom name and type for all atoms
            resid: int  
            output: list of tuples, [(atom1_name, atom1_type), ...]
        """
        res = self.universe.select_atoms(f'protein and resid {resid}')
        atom_names = [(atom.name, atom.type) for atom in res]
        return atom_names
    
    def get_dihedral(self, resid, atom1_name, atom2_name, atom3_name, atom4_name):
        """
            for a specifc diheral angles of a specific residue

            resid: int; 
            atom1_name, atom2_name, atom3_name, atom4_name: name of atoms based on the given psf
        """
        ags = [ self.universe.select_atoms(f"resid {resid} and name {atom1_name} {atom2_name} {atom3_name} {atom4_name}")]
        R = Dihedral(ags).run()
        return R.results.angles
    
    # MDAnalysis provides us with Dihedral module:
    # https://docs.mdanalysis.org/1.1.0/documentation_pages/analysis/dihedrals.html
    def get_phi(self, res_selection=None):
        """
        by default, the first residue has no phi;
        All residue phi angles will be calculated if res_selection is not given.
            res_selection: (could be str) "5-10" means residue index from 5 to 10 will be calculated.
        """
        if res_selection:
            selection = f"protein and resid {res_selection}"
            r = self.universe.select_atoms(selection)
            ags = [res.phi_selection() for res in r.residues]
            R = Dihedral(ags).run()
        else:
            selection = "protein"    # res_selection = "5-10"
            r = self.universe.select_atoms(selection)
            ags = [res.phi_selection() for res in r.residues[1:]]
            R = Dihedral(ags).run()
        return R.results.angles

    def get_psi(self, res_selection=None):
        """
        by default, the last residue has no psi;
        All residue psi angles will be calculated if res_selection is not given.
            res_selection: (could be str) "5-10" means residue index from 5 to 10 will be calculated.
        """
        if res_selection:
            selection = f"protein and resid {res_selection}"
            r = self.universe.select_atoms(selection)
            ags = [res.psi_selection() for res in r.residues]
            R = Dihedral(ags).run()
        else:
            selection = "protein"    # res_selection = "5-10"
            r = self.universe.select_atoms(selection)    
            ags = [res.psi_selection() for res in r.residues[:-1]]
            R = Dihedral(ags).run()
        return R.results.angles

    def get_omega(self, res_selection=None):
        """
        by default, the last residue has no psi;
        All residue omega angles will be calculated if res_selection is not given.
            res_selection: (could be str) "5-10" means residue index from 5 to 10 will be calculated.
        """
        if res_selection:
            selection = f"protein and resid {res_selection}"
            r = self.universe.select_atoms(selection)
            ags = [res.omega_selection() for res in r.residues]
            R = Dihedral(ags).run()
        else:
            selection = "protein"    # res_selection = "5-10"
            r = self.universe.select_atoms(selection) 
            ags = [res.omega_selection() for res in r.residues[:-1]]
            R = Dihedral(ags).run()
        return R.results.angles

    def get_chi1(self, res_selection=None):
        """
        All residue chi1 angles will be calculated if res_selection is not given.
            res_selection: (could be str) "5-10" means residue index from 5 to 10 will be calculated.
        """
        if res_selection:
            selection = f"protein and resid {res_selection} and not (resname GLY ALA)"
        else:
            selection = "protein and not (resname GLY ALA)"    # res_selection = "5-10"
        r = self.universe.select_atoms(selection) 
        ags = [res.chi1_selection() for res in r.residues]
        R = Dihedral(ags).run()
        return R.results.angles
    
    def get_bb_impropers(self, res_selection=None, n_threads=None):
        """
        All residue backbone improper dihedrals will be calculated if res_selection is not given.
            res_selection: (could be str) "5-10" means residue index from 5 to 10 will be calculated.
        """
        if res_selection:
            selection = f"protein and resid {res_selection}"
        else:
            selection = "protein"    # res_selection = "5-10"

        run_per_frame = partial(bb_impropers_per_frame,
                        atomgroup=self.universe.select_atoms(selection))
        
        if n_threads:
            self.available_threads = self.get_available_threads()
        else:
            self.available_threads = n_threads
        
        with Pool(self.available_threads) as worker_pool:
            result = worker_pool.map(run_per_frame, np.arange(self.n_frames))
        impropers = np.asarray(result)
        return impropers


# from MDanalysis: https://userguide.mdanalysis.org/stable/examples/analysis/custom_parallel_analysis.html
def radgyr_per_frame(frame_index, atomgroup, masses):
    # index the trajectory to set it to the frame_index frame
    atomgroup.universe.trajectory[frame_index]

    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates-center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/np.sum(masses)
    # square root and return
    return np.sqrt(rog_sq)

# use "DSSP" for secondary structure calculation
# deprecated due to slow calculation speed
def dssp_per_frame(frame_index, atomgroup):
    # index the trajectory to set it to the frame_index frame
    atomgroup.universe.trajectory[frame_index]
    tmp_pdb = "tmp.pdb"
    atomgroup.write(tmp_pdb)
    structure = PDBParser(QUIET=True).get_structure("structure_id", tmp_pdb)
    model = structure[0]
    dssp = DSSP(model, tmp_pdb, dssp='mkdssp')
    resid = []
    ss = []
    for i,j in zip(list(dssp.keys()), list(model.get_residues())):
        # resid.append(dssp[i][0])
        ss.append(dssp[i][2])
        resid.append(j.full_id[-1][1])
    return resid, ss

def end2end_per_frame(frame_index, atomgroup):
    # index the trajectory to set it to the frame_index frame
    atomgroup.universe.trajectory[frame_index]
    # select n-term N atom
    nterm = atomgroup.select_atoms(f'protein and name N')[0]
    # selct c-term C atom
    cterm = atomgroup.select_atoms(f'protein and name C')[-1]  
    r = cterm.position - nterm.position  # end-to-end vector from atom positions
    d = np.linalg.norm(r)   # end-to-end distance
    return d

def bonds_per_frame(frame_index, atomgroup, atom1_name, atom2_name):
    """
    atomgroup: atomgroup from MDAnalysis
    atom1_name: atom name of the first atom
    atom2_name: atom name of the second atom
    return: bond length array (unit: Angstrom)
    """
    atomgroup.universe.trajectory[frame_index]
    bonds = [bond.value() for bond in atomgroup.bonds if (bond.atoms[0].name == atom1_name and bond.atoms[1].name == atom2_name) or \
                                                         (bond.atoms[0].name == atom2_name and bond.atoms[1].name == atom1_name)]
    return np.array(bonds)

def angles_per_frame(frame_index, atomgroup, atom1_name, atom2_name, atom3_name):
    """
    atomgroup: atomgroup from MDAnalysis
    atom1_name: atom name of the first atom
    atom2_name: atom name of the second atom
    atom3_name: atom name of the third atom
    return: angle array (unit: degree)
    """
    atomgroup.universe.trajectory[frame_index]
    angles = [angle.value() for angle in atomgroup.angles \
              if (angle.atoms[0].name == atom1_name and angle.atoms[1].name == atom2_name and angle.atoms[2].name == atom3_name) or \
                (angle.atoms[0].name == atom3_name and angle.atoms[1].name == atom2_name and angle.atoms[2].name == atom1_name)]
    return np.array(angles)

def dihedrals_per_frame(frame_index, atomgroup, atom1_name, atom2_name, atom3_name, atom4_name):
    """
    atomgroup: atomgroup from MDAnalysis
    atom1_name: atomname of the first atom
    atom2_name: atomname of the second atom
    atom3_name: atomname of the third atom
    atom4_name: atomname of the fourth atom
    return: dihedral array (unit: degree)
    """
    atomgroup.universe.trajectory[frame_index]
    dihedral = [dihedral.value() for dihedral in atomgroup.dihedrals \
              if (dihedral.atoms[0].name == atom1_name and dihedral.atoms[1].name == atom2_name \
                  and dihedral.atoms[2].name == atom3_name and dihedral.atoms[3].name == atom4_name)]
    return np.array(dihedral)

def bb_impropers_per_frame(frame_index, atomgroup):
    """
    atomgroup: atomgroup from MDAnalysis
    return: dihedral array (unit: degree)

    backbone dihedrals: (two for each residue)
        C CA N O
        N C CA H
    """
    atomgroup.universe.trajectory[frame_index]
    bb_impropers = [ improper.value() for improper in atomgroup.impropers if \
                 (improper.atoms[0].type == "C" and improper.atoms[1].type == "CT1" and \
                  improper.atoms[2].type == "NH1" and improper.atoms[3].type == "O") or \
                  (improper.atoms[0].type == "NH1" and improper.atoms[1].type == "C" and \
                  improper.atoms[2].type == "CT1" and improper.atoms[3].type == "H") \
                    ]
    return bb_impropers

    

if __name__ == "__main__":
    pdb = "/home2/jianhuang/projects/VAE/dataset/protein_A/step4.1_equilibration.pro.pdb"
    psf = "/home2/jianhuang/projects/VAE/dataset/protein_A/step1_pdbreader.psf"
    xtc = "/home2/jianhuang/projects/VAE/dataset/protein_A/100ns_gmx_pro_alignca.xtc"
    top = "/home2/jianhuang/projects/VAE/dataset/protein_A/topol.top"

    align_selection = "protein and (resid 10 to 18 or resid 25 to 33 or resid 41 to 55) and name CA"
    rmsd_selection = ["protein and name CA",
                    "protein and (resid 1 to 9) and name CA",
                    "protein and (resid 19 to 24) and name CA",
                    "protein and (resid 34 to 40) and name CA",
                    "protein and (resid 55 to 60) and name CA"]
    
    ensemble_test = Ensemble(psf, xtc, top)

    # RMSF
    # print(ensemble_test.rmsf())

    # Rg
    # ensemble_test.rg()
    # print(ensemble_test.rg)

    # SS: this could be slow...
    # second = ensemble_test.get_ss()

    # get resid and requence
    # print(ensemble_test.sequence)
    # print(ensemble_test.resid)

    # end2end
    # print(ensemble_test.get_end2end())

    # PCA
    # print(ensemble_test.pca(selection="backbone"))

    # bonds
    # bonds = ensemble_test.get_bonds(atom1_type='C', atom2_type='CT1')
    # print(bonds)
    # print(bonds.shape)

    # angles
    # angles = ensemble_test.get_angles(atom1_type='C', atom2_type='CT1', atom3_type='CT3')
    # print(angles)
    # print(angles.shape)

    # dihedrals
    # dihedrals = ensemble_test.get_dihedrals(atom1_type='C', atom2_type='NH1', atom3_type='CT1', atom4_type="C")
    # print(dihedrals)
    # print(dihedrals.shape)

    # get psi, phi and omega
    # psis = ensemble_test.get_psi(res_selection="5-10")
    # print("psis: ", psis)
    # print(psis.shape)
    # phis = ensemble_test.get_phi(res_selection="5-10")
    # print("phis: ", psis)
    # print(phis.shape)
    # omegas = ensemble_test.get_omega(res_selection="5-10")
    # print("omegas: ", psis)
    # print(omegas.shape)

    # get impropers
    impropers = ensemble_test.get_bb_impropers(res_selection="5-10")
    print(impropers)
    print(impropers.shape)

