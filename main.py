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
import sys

# Global variables
# raw_trajectory
traj = sys.args[1]
psf = sys.args[2]


# flags
# flag1: generate input arrary: store as npy file format
# flag2: model traning


def main():
    # extract protein from raw trajectory
    out_psf, out_traj = extract_pro(psf, traj)
    aligned_traj = out_traj.split('.')[0] + '_aligned.xtc'
    traj_align_onfly(out_psf, out_traj, aligned_traj)

    # generate input array
    Ec, bonds, angles, dihedrals = get_ic(out_psf, aligned_traj)
    torsion_scaler, torsion_test, torsion_train = scaling_spliting(dihedrals)


    # define hyperparameter grid for searching
    params_dict = {
            'BATCH_SIZE': 
            'LATENT_DIM': 
            'NUM_HIDDEN_LAYER':
            'EPOCHS':
            'RATE':
            }
    hyperparams_combinations = gen_parms_combinations(**params_dict)


    # VAE model traning
    # some functions here
    for hyperparams_dict in hyperparams_combinations:
        training(**hyperparams_dict)
        # some plot

    # VAE model evaluation
    # dict to store RMSD and correlation;
    # generate the PDB file

    # integrate all PDB files into a xtc
    

    # plot

    pass

