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

aligned_traj = sys.argv[1]
out_psf = sys.argv[2]

# extract protein from raw trajectory
#out_psf, out_traj = extract_pro(psf, traj)
#aligned_traj = out_traj.split('.')[0] + '_aligned.xtc'
#traj_align_onfly(out_psf, out_traj, aligned_traj)

# generate input array
Ec, bonds, angles, dihedrals, R = get_ic(out_psf, aligned_traj)
torsion_scaler, torsion_test, torsion_train = scaling_spliting(dihedrals)


# define hyperparameter grid for searching
params_dict = {
        'BATCH_SIZE': 100,
        'LATENT_DIM': 2,
        'NUM_HIDDEN_LAYER': 3,
        'EPOCHS': 10000,
        'RATE': 0.0001,
        'scaler': torsion_scaler,
        'x_train': torsion_train,
        'x_test': torsion_test
        }

print("HERE")
# VAE model traning
# some functions here
Summary = []
print("here")
return_dict = training(**params_dict)
demap = return_dict['demap']
pdb_temp = '/home/shrishti/Documents/Projects/IDP_ensemble/protein-VAE-main_org/charmm-gui_3GB1/gromacs/t0.pdb'
demap_to_PDB(bonds, angles, Ec, demap,pdb_temp, R)
pickle_file = "pickle_file.pkl"
PDB_to_XTC(pickle_file, pdb_temp)
Summary.append(return_dict)
pickle.dump(summary, open("./summary.pkl", "wb"))

