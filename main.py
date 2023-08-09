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
import json 

def main(traj, psf):

    # parse user-defined variables
    parser = argparse.ArgumentParser(description="VAE model for protein dynamics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--trj", help="raw traj file (format: xtc or dcd)", required=True)
    # parser.add_argument("--psf", help="psf file that is consistent with the providing xtc (format: psf)", required=True)
    # parser.add_argument("--datapath", help="datafile path for trajectory, psf and VAE output (format: str)", required=True)
    # parser.add_argument("--hyperparams", help="hyperparameter search or a given hyperparams set (format: json)", required=True)
    parser.add_argument("--input", help="input json (format: json)", required=True)
    args = parser.parse_args()
    
    # arguments and use absolute path
    with open(args.input, 'r') as f:
        input_args = json.load(f)

    data_path = input_args['datapath']
    traj = os.path.join(data_path, input_args['trj'])
    psf = os.path.join(data_path, input_args['psf'])
    hyperparams_dict = {'BATCH_SIZE': input_args['BATCH_SIZE'], # give a 'list' type
                        'LATENT_DIM': input_args['LATENT_DIM'], #  give a 'list' type
                        'NUM_HIDDEN_LAYER': input_args['NUM_HIDDEN_LAYER'], # give a 'list' type
                        'EPOCHS': input_args['EPOCHS'], # give a 'list' type
                        'RATE': input_args['RATE'], # give a 'list' type
                        }

    # extract protein from raw trajectory
    out_psf, out_traj = extract_pro(psf, traj)
    outtraj_basename = os.path.basename(out_traj)
    outtraj_dirname = os.path.dirname(out_traj)
    aligned_traj = os.path.join(outtraj_dirname , outtraj_basename.split('.')[0] + '_aligned.xtc')
    traj_align_onfly(out_psf, out_traj, aligned_traj)

    # generate input array
    Ec, bonds, angles, dihedrals, R = get_ic(out_psf, aligned_traj)
    torsion_scaler, torsion_test, torsion_train = scaling_spliting(dihedrals)

    # define hyperparameter grid for searching
    hyperparams_combinations = gen_parms_combinations(**hyperparams_dict)

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

        # VAE model evaluation
        return_dict = training(**training_input)
        # dict to store RMSD and correlation;
        Summary.append(return_dict)
        # Save Dictionary
        # some plot
    pickle.dump(summary, open("./summary.pkl", "wb"))

    # generate the PDB file
#    demap_to_PDB(Ic_bonds, Ic_angles, Ec, torsion, R)

    # Generate XTC from PDBs
#    PDB_to_XTC(pickle_file, template_file)
  

    # plot

    return None


main()
