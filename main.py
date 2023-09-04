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
from utils.post_analysis import *
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
    print("aligned_traj",aligned_traj)
    if os.path.isfile(aligned_traj) == False:
        traj_align_onfly(out_psf, out_traj, aligned_traj)

    # generate input array
    if input_args['input_type'] == "dihedral_all":
        Ec, bonds, angles, dihedrals, R = get_ic(out_psf, aligned_traj)
        scaler, test, train, val = scaling_spliting_dihedrals(dihedrals)
        
    elif input_args['input_type'] == "dihedral_backbone":
        original = "original"
        phi, psi = get_bbtorsion(psf, traj)
        dihedrals = np.concatenate((phi,psi),axis=1)
        dihedrals = dihedrals*(np.pi/180)
        Ramachandran_plot_trj(psf, traj, outtraj_dirname)
        scaler, test, train, val = scaling_spliting_dihedrals(dihedrals)
#        phi_plot(phi, outtraj_dirname, original)
#        psi_plot(psi, outtraj_dirname, original)

    elif input_args['input_type'] == "cartesian":
        print("deal with cartesian coordinates")
        coordinates = get_xyz(out_psf, aligned_traj)
        print(coordinates)
        scaler, test, train, val = scaling_spliting_cartesian(coordinates)
        print("scaler")

    elif input_args['input_type'] == "contact_map":
        contact_map = get_contact_map(psf, traj)
        scaler, test, train, val = scaling_spliting_contact_map(contact_map)

    # additional params
    early_stopping = input_args['early_stopping']
    post_analysis = input_args['post_analysis']

    # VAE model traning
    Summary = []
    for hyperparams_dict in hyperparams_combinations:

        # create train input parameters dict
        training_input = hyperparams_dict.copy()
        training_input['scaler'] = scaler
        training_input['x_train'] = train
        training_input['x_test'] = test
        training_input['x_val'] = val
        training_input['early_stopping'] = early_stopping
        training_input['seed'] = seed
        training_input['outtraj_dirname'] = outtraj_dirname

        # VAE model evaluation
        return_dict = training(**training_input)
        # dict to store RMSD and correlation;
        Summary.append(return_dict)

    # Save Dictionary
    pickle.dump(Summary, open(f"{outtraj_dirname}/summary.pkl", "wb"))

    # generate the PDB file and further analysis
    if post_analysis == True:
        # original trajectory analysis 
        post_analysis(summary, input_args['input_type'],psf, xtc, outtraj_dirname, R, template_file)

    return None

main()
