from utils.traj_process import *
from utils.plot import *
from utils.input_gen import *
from utils.train_and_eval import *
import pickle 

def post_analysis(summary, input_args, psf, xtc, outtraj_dirname, R, template_file):
    """
    To do the analysis of true and decoded trajectory
    Contains: RMSD, RMSF, Radius of gyration, secondary structure, Ramachandran plot
    """
    original_rmsd = traj_rmsd(psf, xtc)
    c_alphas.resids, original_rmsf = traj_rmsf(psf, xtc)
    original_rg = traj_rg(psf, xtc)
    original_residues, original_helicity_ave, original_sheet_ave = traj_ss(psf, xtc)
    # Plot all the above properties
    Post_training_analysis_plot(original_rmsd, c_alphas.resids, original_rmsf, original_rg, original_residues, original_helicity_ave, original_sheet_ave, f"{outtraj_dirname}/original")

    # Plot RMSD, Pearsons, and Spearmann
    Testing_analysis_plot(summary, outtraj_dirname)

    for model in Summary:
        # create dir
        hype = model["hyper_together"]
        new_fold = f"{outtraj_dirname}/{hype}"
        path = os.path.join(outtraj_dirname, hype)
        os.mkdir(path)
        stor_fold = f"{hype}/{input_args}"
        path1 = os.path.join(outtraj_dirname, f"{hype}/{input_args}")
        os.mkdir(path1)
        
 
        if input_args == "dihederal_all":
            demap = model["demap"]
            sin_demap = demap[:,:int(len(demap[0])/2)]
            cos_demap = demap[:,int(len(demap[0])/2):]
            demap_rad = np.arctan2(sin_demap, cos_demap)
            pickle_file, new_fold = dihedral_demap_to_PDB(bonds, angles, Ec, demap, pdb, stor_fold, R)
 
        elif input_args == "dihederal_backbone":
            demap = model["demap"]
            sin_demap = demap[:,:int(len(demap[0])/2)]
            cos_demap = demap[:,int(len(demap[0])/2):]
            demap_rad = np.arctan2(sin_demap, cos_demap)
            demap_degrees = demap*(180/np.pi) 
            phi_demap = demap_degrees[:,:int(len(demap_degrees[0])/2)]
            psi_demap = demap_degrees[:, int(len(demap_degrees[0])/2):]
 
            phi_comp_plot(phi, phi_demap, new_fold)
            psi_comp_plot(psi, psi_demap, new_fold)
            Ramachandran_plot_decode(phi_demap, psi_demap, new_fold)
            Ramachandran_plot_comp(phi, psi, phi_demap, psi_demap, stor_fold)
                    
        elif input_args == "cartesian":
            demap = model["demap"]
            pickle_file, new_fold = cartesian_demap_to_PDB(demap, new_fold)
      
        # Convert multiple PDBs to single XTC file    
        PDB_to_XTC(pickle_file, template_file, new_fold)
        out_xtc = f"{new_fold}/output_xtc.xtc"
        out_psf, decoded_traj = traj_align_onfly(psf, traj, out_xtc)
        rmsd_matrix = traj_rmsd(psf, decoded_traj)
        c_alphas, rmsf_matrix = traj_rmsf(psf, decoded_traj)
        Rgyr = traj_rg(psf, decoded_traj)
        residues, helicity_ave, sheet_ave = traj_ss(psf, decoded_traj)
        Post_training_analysis_plot(rmsd, c_alphas, rmsf, Rgyr, residues, helicity_ave, sheet_ave, out_path)

    return None
