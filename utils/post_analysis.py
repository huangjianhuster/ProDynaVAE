from utils.traj_process import *
from utils.plot import *
from utils.input_gen import *
from utils.train_and_eval import *
import pickle 

def Post_Analysis(summary, input_args, psf, xtc, outtraj_dirname,  pdb,timestep, rmsd_names, rmsd_cal, align_select):
    """
    To do the analysis of true and decoded trajectory
    Contains: RMSD, RMSF, Radius of gyration, secondary structure, Ramachandran plot
    """
    original_rmsd = traj_rmsd(psf, xtc, align_select, rmsd_cal)
    resids, original_rmsf = traj_rmsf(psf, xtc)
    original_rg = traj_rg(psf, xtc)
    original_helicity_ave, original_sheet_ave = traj_ss(psf, xtc)
    original_pc = PCA(psf,xtc) 
    original_ete = endtoend(psf,xtc)

    # Plot all the above properties
    orgi_fold = f"{outtraj_dirname}/original"
    if os.path.exists(orgi_fold) == False:
        path = os.path.join(orgi_fold)
        os.mkdir(path)
    Post_training_analysis_plot(original_rmsd, resids, original_rmsf, original_rg, original_helicity_ave, original_sheet_ave, orgi_fold,"Original",rmsd_names)
 
    # Plot RMSD, Pearsons, and Spearmann
    Testing_analysis_plot(summary, outtraj_dirname)

    for model in summary:
        # create dir
        hype = model["hyper_together"]
        new_fold = f"{outtraj_dirname}/{hype}"
        demap = model["demap"]
        x_train = model["train"]
        x_test = model["test"]
        if os.path.exists(new_fold) == False:
            path = os.path.join(outtraj_dirname, hype)
            os.mkdir(path)
        
        if input_args == "dihederal_all":
            sin_demap = demap[:,:int(len(demap[0])/2)]
            cos_demap = demap[:,int(len(demap[0])/2):]
            demap_rad = np.arctan2(sin_demap, cos_demap)
            pickle_file, new_fold = dihedral_demap_to_PDB(bonds, angles, Ec, demap, pdb, stor_fold, R)
 
        elif input_args == "dihederal_backbone":
            sin_demap = demap[:,:int(len(demap[0])/2)]
            cos_demap = demap[:,int(len(demap[0])/2):]
            demap_rad = np.arctan2(sin_demap, cos_demap)
            demap_degrees = demap*(180/np.pi) 
            phi_demap = demap_degrees[:,:int(len(demap_degrees[0])/2)]
            psi_demap = demap_degrees[:, int(len(demap_degrees[0])/2):]
 
            phi_comp_plot(phi, phi_demap, new_fold,hype)
            psi_comp_plot(psi, psi_demap, new_fold,hype)
            Ramachandran_plot_decode(phi_demap, psi_demap, new_fold, hype)
            Ramachandran_plot_comp(phi, psi, phi_demap, psi_demap, stor_fold, hype)
                    
        elif input_args == "cartesian":
            split_pdb = pdb.split(".")
            spl_pdb = split_pdb[0].split("/")
            template_file = f"{new_fold}/{spl_pdb[-1]}_heavy.pdb"
            template_gro = f"{new_fold}/{spl_pdb[-1]}_heavy.gro"
            if os.path.isfile(template_file) == False:
                heavy_atom_templete(pdb,template_file,template_gro)
            pdbs = f"{new_fold}/pdb"
            if os.path.exists(pdbs) == False:
                path = os.path.join(pdbs)
                os.mkdir(pdbs)
            pickle_file = cartesian_demap_to_PDB(demap, pdbs, template_file)
            pdbs_test = f"{new_fold}/pdb_test"
            if os.path.exists(pdbs_test) == False:
                path = os.path.join(pdbs_test)
                os.mkdir(pdbs_test)
            pickle_file_test = cartesian_demap_to_PDB(x_test, pdbs_test, template_file)
      
        elif input_args == "calpha":
            split_pdb = pdb.split(".")
            spl_pdb = split_pdb[0].split("/")
            template_file = f"{new_fold}{spl_pdb[-1]}_CA.pdb"
            template_gro = f"{new_fold}{spl_pdb[-1]}_CA.gro"
            if os.path.isfile(template_file) == False:
                CA_atom_templete(pdb,template_file,template_gro)
            pdbs = f"{new_fold}/pdb"
            if os.path.exists(pdbs) == False:
                path = os.path.join(pdbs)
                os.mkdir(pdbs)
            pickle_file1 = cartesian_demap_to_PDB(demap, pdbs, template_file)
            pickle_file = CAtoFull(pickle_file1, pdbs)
            pdbs_test = f"{new_fold}/pdb_test"
            if os.path.exists(pdbs_test) == False:
                path = os.path.join(pdbs_test)
                os.mkdir(pdbs_test)
            pickle_file_test1 = cartesian_demap_to_PDB(x_test, pdbs_test, template_file)
            pickle_file_test = CAtoFull(pickle_file_test1, pdbs_test)


        # Convert multiple PDBs to single XTC file    
        PDB_to_XTC(pickle_file, template_file, pdbs, timestep)
        PDB_to_XTC(pickle_file_test, template_file, pdbs_test, timestep)
        traj = f"{pdbs}/output_xtc.xtc"
        out_xtc = f"{pdbs}/output_xtc_aligned.xtc"
        traj_align_onfly(template_gro, traj, out_xtc, center=False)
        test_traj = f"{pdbs_test}/output_xtc.xtc"
        test_xtc = f"{pdbs_test}/output_xtc_aligned.xtc"
        traj_align_onfly(template_gro, test_traj, test_xtc, center=False)

        rmsd_matrix = traj_rmsd(template_gro, out_xtc, align_select, rmsd_cal)
        new_rmsd = original_rmsd[1][:len(rmsd_matrix[1])]
        rmsd_matrix[1] = new_rmsd
        c_alphas, rmsf_matrix = traj_rmsf(template_gro, out_xtc)
        Rgyr = traj_rg(template_gro, out_xtc)
        helicity_ave, sheet_ave = traj_ss(template_gro, out_xtc)
        pc = PCA(psf,xtc)     
        ete = endtoend(psf,xtc)
        Post_training_analysis_plot(rmsd_matrix, c_alphas, rmsf_matrix, Rgyr, helicity_ave, sheet_ave, new_fold, hype,rmsd_names)

        # Test original
         
        # Comparision 
        rmsd_test = traj_rmsd(template_gro, test_xtc, align_select, rmsd_cal)
        c_alphas_test, rmsf_test = traj_rmsf(template_gro, test_xtc)
        Rgyr_test = traj_rg(template_gro, test_xtc)
        helicity_ave_test, sheet_ave_test = traj_ss(template_gro, test_xtc)

        Post_training_analysis_plot_comp(rmsd_test, c_alphas, rmsf_test, Rgyr_test, helicity_ave_test, sheet_ave_test, new_fold, hype, rmsd_matrix, c_alphas, rmsf_matrix, Rgyr, helicity_ave, sheet_ave,rmsd_names)

    return None
