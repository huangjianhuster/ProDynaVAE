# Author: Jian Huang
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# TODO:
# integrate the whole VAE training process

# Dependencies
# MDAnalysis
# MDTraj

from utils.traj_process import *
from utils.plot import *
from utils.input_gen import *
from utils.train_and_eval import *
import argparse
import sys
import os
from tensorflow import random
import json 

def main():
    # parse user-defined variables
    parser = argparse.ArgumentParser(description="VAE model for protein dynamics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", help="input json (format: json)", required=True)
    args = parser.parse_args()
    
    # arguments and use absolute path
    with open(args.input, 'r') as f:
        input_args = json.load(f)


    seed = input_args['seed'] 
    data_path = input_args['datapath']
    traj = os.path.join(data_path, input_args['trj'])
    psf = os.path.join(data_path, input_args['psf'])
    pdb = os.path.join(data_path, input_args['pdb'])
    
    hyperparams_dict = {'BATCH_SIZE': input_args['BATCH_SIZE'], # give a 'list' type
                        'LATENT_DIM': input_args['LATENT_DIM'], #  give a 'list' type
                        'NUM_HIDDEN_LAYER': input_args['NUM_HIDDEN_LAYER'], # give a 'list' type
                        'EPOCHS': input_args['EPOCHS'], # give a 'list' type
                        'RATE': input_args['RATE'], # give a 'list' type
                        }
    hyperparams_combinations = gen_parms_combinations(**hyperparams_dict)
    print(hyperparams_combinations)

    random.set_seed(seed)
    # extract protein from raw trajectory
    out_psf, out_traj = extract_pro(psf, traj)
    outtraj_basename = os.path.basename(out_traj)
    outtraj_dirname = os.path.dirname(out_traj)
    aligned_traj = os.path.join(outtraj_dirname , outtraj_basename.split('.')[0] + '_aligned.xtc')
    traj_align_onfly(out_psf, out_traj, aligned_traj)

    # generate input array
    Ec, bonds, angles, dihedrals, R = get_ic(out_psf, aligned_traj)
    torsion_scaler, torsion_test, torsion_train = scaling_spliting(dihedrals)
    
    # additional params
    early_stopping = input_args['early_stopping']
    post_analysis = input_args['post_analysis']

    print("HERE")
    # VAE model traning
    # some functions here
    Summary = []
    for hyperparams_dict in hyperparams_combinations:
        print("here")

        # create train input parameters dict
        training_input = hyperparams_dict.copy()
        training_input['scaler'] = torsion_scaler
        training_input['x_train'] = torsion_train
        training_input['x_test'] = torsion_test
        training_input['early_stopping'] = early_stopping
        training_input['seed'] = seed
        training_input['outtraj_dirname'] = outtraj_dirname

        # VAE model evaluation
        return_dict = training(**training_input)
        # dict to store RMSD and correlation;
        Summary.append(return_dict)
        # Save Dictionary
        # some plot
    pickle.dump(Summary, open(f"{outtraj_dirname}/summary.pkl", "wb"))

    # generate the PDB file
    if post_analysis == True:
      
        demap = return_dict["demap"] 
        demap_to_PDB(bonds, angles, Ec, demap, pdb, outtraj_dirname, R)
        PDB_to_XTC(pickle_file, pdb, outtraj_dirname)
        #pickle_file = "pickle_file.pkl"
        
        # Plot RMSD, Pearsons, and Spearmann
        Testing_analysis_plot(summary, outtraj_dirname)
        out_xtc = f"{outtraj_dirname}/out_xtc.xtc"
        out_psf, decoded_traj = traj_align_onfly(psf, xtc, out_xtc)
        rmsd_matrix = traj_rmsd(psf, out_xtc)
        c_alphas.resids, rmsf_matrix = traj_rmsf(psf, out_xtc)
        Rgyr = traj_rg(psf, out_xtc)
        residues, helicity_ave, sheet_ave = traj_ss(psf, out_xtc)

       
        Post_training_analysis_plot(rmsd, c_resids, rmsf, Rgyr, residues, helicity_ave, sheet_ave, out_path)

    return None


main()
