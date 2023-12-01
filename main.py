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

def get_input(psf, pdb, traj, split, input_type="cartesian"):
    """
        -- Convert trajectory into input for VAE model
    """
    R = None
    if input_type == "dihedral_all":
        remove_selection = None
        Ec, bonds, angles, dihedrals, R = get_ic(psf, traj)
        scaler, test, train = scaling_spliting_dihedrals(dihedrals)
        
    elif input_type == "dihedral_backbone":
        remove_selection = None
        original = "original"
        phi, psi = get_bbtorsion(psf, traj)
        dihedrals = np.concatenate((phi,psi),axis=1)
        dihedrals = dihedrals*(np.pi/180)
        scaler, test, train = scaling_spliting_dihedrals(dihedrals)

    elif input_type == "cartesian":
        remove_selection = "not name H*"
        coordinates = get_xyz(pdb, traj)
        scaler, test, train = scaling_spliting_cartesian(coordinates, split)

    elif input_type == "calpha":
        remove_selection = "name CA and not protein"
        coordinates = get_cxyz(pdb, traj)
        scaler, test, train = scaling_spliting_cartesian(coordinates , split)

    elif input_type == "contact_map":
        remove_selection = None
        contact_map = get_contact_map(psf, traj)
        scaler, test, train = scaling_spliting_contact_map(contact_map, split)
    
    return scaler, test, train, R, remove_selection

def main():
    # parse user-defined variables
    parser = argparse.ArgumentParser(description="VAE model for protein dynamics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", help="input json (format: json)", required=True)
    args = parser.parse_args()
    
    # arguments and use absolute path
    # load parameters
    with open(args.input, 'r') as f:
        input_args = json.load(f)
    seed = input_args['seed']   # random seed
    data_path = input_args['datapath']
    traj = input_args['trj']
    psf = input_args['psf']
    pdb = input_args['pdb']
    hyperparams_dict = {'BATCH_SIZE': input_args['BATCH_SIZE'], # give a 'list' type
                        'LATENT_DIM': input_args['LATENT_DIM'], #  give a 'list' type
                        'NUM_HIDDEN_LAYER': input_args['NUM_HIDDEN_LAYER'], # give a 'list' type
                        'EPOCHS': input_args['EPOCHS'], # give a 'list' type
                        'RATE': input_args['RATE'], # give a 'list' type
                        }
    hyperparams_combinations = gen_parms_combinations(**hyperparams_dict)
    random.set_seed(seed)

    # make dir for decoder-generated trajectories
    outtraj_dirname = f"{data_path}/{input_args['input_type']}"
    if os.path.exists(outtraj_dirname) == False:
        path = os.path.join(outtraj_dirname)
        os.mkdir(path)
    
    # generate input array
    scaler, test, train, R, remove_selection = get_input(psf, traj, input_args['split'], input_type=input_args['input_type'])
    # add argument to save the train, validation, test
    picle.dump(test, open(f"{outtraj_dirname}_test_{seed}.pkl", "wb"))
    picle.dump(train, open(f"{outtraj_dirname}_train_{seed}.pkl", "wb"))

    # VAE model traning
    Summary = []
    for hyperparams_dict in hyperparams_combinations:
        # create train input parameters dict for the "training" function
        training_input = hyperparams_dict.copy()
        # load other training details: training data, testing data
        training_input['scaler'] = scaler
        training_input['x_train'] = train
        training_input['x_test'] = test
        training_input['split'] = input_args['split']
        training_input['early_stopping'] = input_args['early_stopping']
        training_input['seed'] = input_args['seed']
        training_input['outtraj_dirname'] = outtraj_dirname
        # VAE model evaluation
        return_dict = training(**training_input)
        # generate decoder xtc files
        demap_to_xtc(psf, return_dict['demap'], remove_selection, f"{outtraj_dirname}/{return_dict['hyper_together']}/decoder.xtc")
        Summary.append(return_dict)
        # dict to store RMSD and correlation;
        del return_dict['outtraj_dirname']
        del return_dict['demap']
        del return_dict['hyper_together']
        Save_summary.append(return_dict)

    # Save Dictionary
    Save_summary.to_csv(f"{outtraj_dirname}/summary.pkl",index=False)

    # generate the PDB file and further analysis
    if input_args['post_analysis'] == True:
        
        # original trajectory analysis 
        # Post_Analysis(Summary,input_args['input_bad'], input_args['input_type'], psf, traj, input_args['top'], outtraj_dirname, pdb, input_args['timestep'], input_args['rmsd_names'],input_args['rmsd_cal'],input_args['selection'])
        pass
    print("job finished")
    return None

if __name__ == "__main__":
    main()
