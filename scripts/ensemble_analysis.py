import sys
sys.path.append("../../ProDynaVAE")
from utils.ensemble import *

# TODO: an exmple of using utils.ensemble module

# Internal parameters
def get_dihedrals(ensemble, resid):
    """
    Get the general dihedrals: phi, psi, omega and chi1 from an ensemble

    ensemble: utils.Ensemble objects
    resid: str, "5"
    """
    phi = ensemble.get_phi(res_selection=resid)
    psi = ensemble.get_psi(res_selection=resid)
    omega = ensemble.get_omega(res_selection=resid)
    chi1 = ensemble.get_chi1(res_selection=resid)
    return phi, psi, omega, chi1

def compare_dihedrals(ensemble1, ensemble2, resid, binwidth=5):
    """
    compare dihedral distributions between two ensembles, including phi, psi, omega, chi1
    """
    phi_1, psi_1, omega_1, chi1_1 = get_dihedrals(ensemble=ensemble1, resid=resid)
    phi_2, psi_2, omega_2, chi1_2 = get_dihedrals(ensemble=ensemble2, resid=resid)

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    axs = axs.ravel()

    for ax, dihe1, dihe2, name in zip(axs, [phi_1, psi_1, omega_1, chi1_1],\
                                       [phi_2, psi_2, omega_2, chi1_2], ['Phi', 'Psi', 'Omega', 'Chi1']):

        ax.hist(dihe1, bins=np.arange(min(dihe1), max(dihe1) + binwidth, binwidth),\
                density=True, color='blue', alpha=0.7, edgecolor='black', label="Ensemble1")
        ax.hist(dihe2, bins=np.arange(min(dihe2), max(dihe2) + binwidth, binwidth),\
                    density=True, color='red', alpha=0.7, edgecolor='black', label="Ensemble2")
        ax.set_title(f"{name} angle distribution")

        ax.set_ylabel("Probability density")
        ax.grid()
        ax.legend(loc="upper left", frameon=False)

    return [phi_1, psi_1, omega_1, chi1_1], [phi_2, psi_2, omega_2, chi1_2], fig, axs

def compare_bond(ensemble1, ensemble2, resid, atom1_name, atom2_name, binwidth=0.05):
    """
    compare bond distributions between two ensembles
    """
    bond_ensem1 = ensemble1.get_bond(selection=f"protein and resid {resid}", \
                                      atom1_name=atom1_name, atom2_name=atom2_name)
    bond_ensem2 = ensemble2.get_bond(selection=f"protein and resid {resid}", \
                                      atom1_name=atom1_name, atom2_name=atom2_name)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(bond_ensem1, bins=np.arange(min(bond_ensem1), max(bond_ensem1) + binwidth, binwidth),\
            density=True, color='blue', alpha=0.7, edgecolor='black', label="Ensemble1")
    ax.hist(bond_ensem2, bins=np.arange(min(bond_ensem2), max(bond_ensem2) + binwidth, binwidth),\
                density=True, color='red', alpha=0.7, edgecolor='black', label="Ensemble2")
    ax.set_title(f"{atom1_name}-{atom2_name} bond distribution")
    ax.set_ylabel("Probability density")
    ax.grid()
    ax.legend(loc="upper left", frameon=False)

    return (bond_ensem1, bond_ensem2), fig, ax

def compare_angle(ensemble1, ensemble2, resid, atom1_name, atom2_name, atom3_name, binwidth=5):
    """
    compare angle distributions between two ensembles
    """
    angle_ensem1 = ensemble1.get_angle(selection=f"protein and resid {resid}", \
                                        atom1_name=atom1_name, atom2_name=atom2_name, atom3_name=atom3_name)
    angle_ensem2 = ensemble2.get_angle(selection=f"protein and resid {resid}", \
                                        atom1_name=atom1_name, atom2_name=atom2_name, atom3_name=atom3_name)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(angle_ensem1, bins=np.arange(min(angle_ensem1), max(angle_ensem1) + binwidth, binwidth),\
            density=True, color='blue', alpha=0.7, edgecolor='black', label="Ensemble1")
    ax.hist(angle_ensem2, bins=np.arange(min(angle_ensem2), max(angle_ensem2) + binwidth, binwidth),\
                density=True, color='red', alpha=0.7, edgecolor='black', label="Ensemble2")
    ax.set_title(f"{atom1_name}-{atom2_name}-{atom3_name} angle distribution")
    ax.set_ylabel("Probability density")
    ax.grid()
    ax.legend(loc="upper left", frameon=False)

    return (angle_ensem1, angle_ensem2), fig, ax

if __name__ == "__main__":
    # MD ensemble
    proG_training_xtc = "/home2/jianhuang/projects/VAE/dataset/protein_G/100ns_pro_align_alphahelix.xtc"
    proG_training_psf = "/home2/jianhuang/projects/VAE/dataset/protein_G/step1_pdbreader.psf"
    proG_training_pdb = "/home2/jianhuang/projects/VAE/dataset/protein_G/step4.1_equilibration.pro.pdb"
    proG_top = "/home2/jianhuang/pikes_home/work/VAE/protein_G/charmm-gui-0110069008/gromacs/topol.top"

    # Decoder ensemble
    proG_decoder_xtc = "/home2/jianhuang/projects/VAE/training/protein_G/cartesian_z/B1000LD6HL1E200R0.001S0.7/decoder.xtc"
    proG_noH_pdb = "/home2/jianhuang/projects/VAE/training/protein_G/cartesian_z/B1000LD6HL1E200R0.001S0.7/test_noH.pdb"
    proG_noH_psf = "/home2/jianhuang/projects/VAE/dataset/protein_G/noH.psf"

    # Create Ensemble objects
    proG_decoder_ensemble = Ensemble(proG_noH_psf, proG_decoder_xtc, proG_top)
    proG_training_ensemble = Ensemble(proG_training_psf, proG_training_xtc, proG_top)

    # Let's use residue 10 (Lysine) in the above case
    # First get a list of all atom names and types of the resid 10
    print(proG_training_ensemble.get_atoms_info(resid=10))

    # compare dihedrals
    dihe_ensem1, dihe_ensem2, (fig_dihe, ax_dihe) = compare_dihedrals(proG_training_ensemble, \
                                                                       proG_decoder_ensemble, resid="10")
    #get eq values from top
    print(proG_training_ensemble.get_dihedral_eq('C', 'NH1', 'CT1', 'C'))   # phi; using atom type!
    print(proG_training_ensemble.get_dihedral_eq('NH1', 'CT1', 'C', 'NH1')) # psi

    # compare bonds
    proG_training_ensemble.get_atoms_info(resid=10)
    (bond_md, bond_decoder), fig_bond, ax_bond = compare_bond(proG_training_ensemble, proG_decoder_ensemble, \
                                                     resid="10", atom1_name="CE", atom2_name="NZ")
    print(proG_training_ensemble.get_bond_eq(atom1_type='CT2', atom2_type='NH3')) # using atom type!

    # compare angles
    (angle_md, angle_decoder), fig_angle, ax_angle = compare_angle(proG_training_ensemble, proG_decoder_ensemble, \
                                                                    resid="10", atom1_name="CD", atom2_name="CE", \
                                                                    atom3_name="NZ", binwidth=5)
    print(proG_training_ensemble.get_angle_eq(atom1_type='CT2', atom2_type='CT2', atom3_type='NH3'))
    
    # compare secondary structure
    helix_ensem1, sheet_ensem1 = proG_training_ensemble.get_ss()
    helix_ensem2, sheet_ensem2 = proG_decoder_ensemble.get_ss()

    # compare end2end
    e2e_ensem1 = proG_training_ensemble.get_end2end()
    e2e_ensem2 = proG_decoder_ensemble.get_end2end()

    # compare backbone ramachandran
    ram_ensem1 = proG_training_ensemble.universe.select_atoms("protein")
    ram_ensem1_R = Ramachandran(ram_ensem1).run()
    ram_ensem1_R.plot()
    ram_ensem2 = proG_decoder_ensemble.universe.select_atoms("protein")
    ram_ensem2_R = Ramachandran(ram_ensem2).run()
    ram_ensem2_R.plot()
