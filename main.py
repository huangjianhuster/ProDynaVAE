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
import sys

# Global variables
# raw_trajectory
traj = sys.argv[1]
psf = sys.argv[2]
#out_path = "/tmp/"
#params_json = sys.argv[3]

# flags
# flag1: generate input arrary: store as npy file format
# flag2: model traning


def main(traj, psf):
    # extract protein from raw trajectory
    out_psf, out_traj = extract_pro(psf, traj)
    aligned_traj = out_traj.split('.')[0] + '_aligned.xtc'
    traj_align_onfly(out_psf, out_traj, aligned_traj)

    # generate input array
    Ec, bonds, angles, dihedrals, R = get_ic(out_psf, aligned_traj)
    torsion_scaler, torsion_test, torsion_train = scaling_spliting(dihedrals)


    # define hyperparameter grid for searching
    params_dict = {
            'BATCH_SIZE': 64,
            'LATENT_DIM': 2,
            'NUM_HIDDEN_LAYER': 4,
            'EPOCHS': 5,
            'RATE': 0.001,
            'scaler': torsion_scaler,
            'x_train': torsion_train,
            'x_test': torsion_test
            }
    hyperparams_combinations = gen_parms_combinations(**params_dict)

    print("HERE")
    # VAE model traning
    # some functions here
    Summary = []
    for hyperparams_dict in hyperparams_combinations:
        print("here")
        # VAE model evaluation
        return_dict = training(**hyperparams_dict)
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

