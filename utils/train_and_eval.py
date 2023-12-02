# Dependencies
import pickle
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import MDAnalysis as mda
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import random
import sys
import datetime
from subprocess import run

import multiprocessing
from multiprocessing import Lock, Process, Queue, current_process

from utils.model import *
from utils.plot import *


def training(**kwargs):
    """
    **kwargs: dict to specify hyperparameters:
        {
            "scaler": scaler,
            "x_test": x_test_array,
            "x_train": x_train_array,

            "BATCH_SIZE": 64,
            "LATENT_DIM": 2,
            "NUM_HIDDEN_LAYER": 4,
            "EPOCHS": 200,
            "RATE": 0.00001,
            "early_stopping": True,
            "seed": 20
        }

    """
    seed = kwargs['seed']
    random.set_seed(seed)
    scaler = kwargs['scaler']
    x_test = kwargs['x_test']
    x_train = kwargs['x_train']
    original_dim = x_train.shape[1]

    BATCH_SIZE = kwargs['BATCH_SIZE']
    LATENT_DIM = kwargs['LATENT_DIM']
    NUM_HIDDEN_LAYER = kwargs['NUM_HIDDEN_LAYER']
    EPOCHS = kwargs['EPOCHS']
    RATE = kwargs['RATE']
    early_stopping = kwargs['early_stopping']
    outtraj_dirname = kwargs['outtraj_dirname']

    # Build VAE MODEL
    encoder, decoder, vae = build_vae(original_dim, LATENT_DIM, NUM_HIDDEN_LAYER, RATE)
    encoder.summary()

    # Train VAE MODEL
    log_dir = str(outtraj_dirname) + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
 
    if "early_stopping" == False:
        history = vae.fit(x=x_train, y=x_train,
                shuffle=True,
                epochs=EPOCHS,
                validation_split=0.2, # validation_data=(x_val, x_val),
                verbose=2,
                callbacks = [tensorboard_callback])
    else:
        # If you want early stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        history = vae.fit(x=x_train, y=x_train,
                shuffle=True,
                epochs=EPOCHS,
                validation_split=0.2, # validation_data=(x_val, x_val),
                verbose=2,
                callbacks=[early_stopping, tensorboard_callback])


    formatter = mpl.ticker.EngFormatter()
    all_hype = f"B{BATCH_SIZE}LD{LATENT_DIM}HL{NUM_HIDDEN_LAYER}E{EPOCHS}R{RATE}S{kwargs['split']}"
    # Plot losses
    plot1 = train_test_loss_plot(history, outtraj_dirname, all_hype)
    # TEST 
    encoded = encoder.predict(x_test, batch_size=BATCH_SIZE)
    decoded = decoder.predict(encoded[0])
    demap =  scaler.inverse_transform(decoded)     
    x_train = scaler.inverse_transform(x_train)
    x_test = scaler.inverse_transform(x_test)
    # Plot here
    #latent_space_plot(encoded, save_path)

    # Evaluation 
    Spearmann, Pearson, pv_spearman, pv_pearson, RMSD_mean, RMSD_std = evaluate(encoded, demap, x_test, scaler) 
    # Create return dict
    return_dict = {}
    
    return_dict['BATCH_SIZE'] = BATCH_SIZE
    return_dict['LATENT_DIM'] = LATENT_DIM
    return_dict['NUM_HIDDEN_LAYER'] = NUM_HIDDEN_LAYER
    return_dict['EPOCHS'] = EPOCHS
    return_dict['RATE'] = RATE

    return_dict['Spearmann'] = Spearmann
    return_dict['Pearson'] = Pearson 
    return_dict['val_spearman'] = pv_spearman
    return_dict['val_pearson'] = pv_pearson
    return_dict['RMSD_mean'] = RMSD_mean
    return_dict['RMSD_std'] = RMSD_std
    return_dict['demap'] = demap
    return_dict['outtraj_dirname'] = outtraj_dirname
    return_dict['hyper_together'] = all_hype
    return return_dict

# Sampling from Latent space
#def sample_latent():

def evaluate(encoded, demap, x_test, scaler):
    Spearmann, Pearson, pv_spearman, pv_pearson = cal_spearman(x_test, encoded[0])
    RMSD_mean, RMSD_std = cal_rmsd(x_test, demap)
    return Spearmann, Pearson, pv_spearman, pv_pearson, RMSD_mean, RMSD_std

# Spearmann correlation and Pearson correlation
def cal_spearman(data_original, data_encoded):
    dist_encoded = np.square(euclidean_distances(data_encoded, data_encoded)).flatten()
    dist_original = np.square(euclidean_distances(data_original, data_original)).flatten()
    spearman, pv_spearman = spearmanr(dist_original, dist_encoded)
    pearson, pv_pearson = pearsonr(dist_original, dist_encoded)
    return spearman, pearson, pv_spearman, pv_pearson

# RMSD in A
def cal_rmsd(data_original, data_decoded):
    """
    Calculate RMSD between input and output array
    Input eg. (test data, output data after inverse transformation, 
    return mean RMSD and standard deviation 
    """
    data_decoded = data_decoded
    a = np.sum(np.square(data_decoded - data_original), axis=1)
    rmsd = np.sqrt(a / (data_original.shape[1]))
    return np.mean(rmsd), np.std(rmsd)

def dihedral_demap_to_PDB(Ic_bonds, Ic_angles, Ec, torsion,pdb_temp, outtraj_dirname,R):
    Ic_bonds_angles_ave = np.average(np.concatenate([Ic_bonds, Ic_angles], axis=1), axis=0)
    new_fold = f"{outtraj_dirname}/pdb"
    if os.path.exists(new_fold) == False:
        path = os.path.join(outtraj_dirname,"pdb")
        os.mkdir(path)
    i = 0
    file = []
    for tors in torsion:
        final_array = np.concatenate([Ec, Ic_bonds_angles_ave, tors])
        xyz = R.Cartesian(final_array)
    
        # Flatten the array
        new_xyz = xyz.reshape(-1)
    
        # Write into PDB
        template = extract_template(pdb_temp)
        coorinates = load_coor(new_xyz,template)
        write_file(f"{new_fold}/{i}.pdb",coorinates)
        file.append(f"{i}.pdb")
        i += 1  
    with open(f"{new_fold}/pickle_file.pkl", 'wb') as w:
        pickle.dump(file, w)
    pick_file = f"{new_fold}/pickle_file.pkl"
    return pick_file, new_fold

def cartesian_demap_to_PDB(demap, outtraj_dirname, pdb_temp):
    i = 0
    file = []
    for x in demap:
        template = extract_template(pdb_temp)
        coorinates = load_coor( x,template)
        write_file(f"{outtraj_dirname}/{i}.pdb",coorinates)
        file.append(f"{i}.pdb")
        i += 1
    with open(f"{outtraj_dirname}/pickle_file.pkl", 'wb') as w:
        pickle.dump(file, w)
    pick_file = f"{outtraj_dirname}/pickle_file.pkl"
    return pick_file

# CA to fully-atomic
def CAtoFull(pick_file, outtraj_dirname):
    pick_file1 = []
    pdb_file = pickle.load(open(f"{pick_file}",'rb'))
    k = 0
    for i in range(0,len(pdb_file),12):
        #sm1 = []
        for j in range(k,12,1): 
            sm = f'./scripts/ModRefiner-l/mcrefinement {outtraj_dirname}  scripts/ModRefiner-l {pdb_file[j]} {pdb_file[j]} 5'.split()
            #sm1.append(f"p{j}")
            sm1 = multiprocessing.Process(target=sm)
            sm1.start()
            sm1.join() 
        emd1 = []
        for j in range(k,12,1):
            emd = f'./scripts/ModRefiner-l/mcrefinement {outtraj_dirname}  scripts/ModRefiner-l {pdb_file[j]} {pdb_file[j]} 100 5'.split()
            emd1.append(f"p{j}")
            emd1 = multiprocessing.Process(emd)
            emd1.start()
            emd1.join()
            pick_file1.append(f"em{pdb_file[j]}")
        k = k + 12
    with open(f"{outtraj_dirname}/pickle_file_new.pkl", 'wb') as w:
        pickle.dump(pdb_file, outtraj_dirname)
    pick_file1 = f"{outtraj_dirname}/pickle_file_new.pkl"
    return pick_file1

def PDB_to_XTC(pickle_file, template_file, output_xtc,time_s):
    """
        Mutiple PDB to XTC conveter 
    """
    pdb_file = pickle.load(open(f"{pickle_file}",'rb'))
    reformat_pdbfiles = [(i) for i in pdb_file]
    u = mda.Universe(template_file)

    xtc_tmp = f"{output_xtc}/output_tmp.xtc"
    xtc_file = f"{output_xtc}/output_xtc.xtc"  
    with mda.Writer(xtc_tmp, len(u.atoms)) as xtc_writer:
        for pdb_file1 in reformat_pdbfiles[0:]:
            u.load_new(f"{output_xtc}/{pdb_file1}")  # Load each PDB file into the Universe
            xtc_writer.write(u)

    # Add time step in the trajectory
    u2 = mda.Universe(template_file, xtc_tmp)
    for ts in u2.trajectory:
        ts.data['dt'] = time_s
        ts.data['time'] = ts.frame*time_s
        ts.data['step'] = ts.frame
    with mda.Writer(xtc_file, u2.atoms.n_atoms) as W:
        for ts in u2.trajectory:
            W.write(u2.atoms)
    for temp_file in pdb_file:
        os.system("rm %s" % f"{output_xtc}/{temp_file}")
    os.system("rm %s" % xtc_tmp)
    return None

def demap_to_xtc(psf, demap, remove_selection, out_xtc):
    u = mda.Universe(psf)
    nonH_atoms = u.select_atoms(remove_selection)   # example: "not name H*"

    num_frames = demap.shape[0]
    
    # make sure dir has been made
    if os.path.exists(out_xtc) == False:
        os.mkdir(out_xtc)

    nonH = mda.Merge(nonH_atoms)
    nonH.load_new(demap[0].reshape((len(nonH_atoms), 3)))
    nonH.select_atoms("all").write(f"{out_xtc}/noH.pdb")

    with mda.Writer(f"{out_xtc}/decoder.xtc", len(nonH.atoms)) as xtc_writer:
        for frame in range(num_frames):
            nonH.load_new(demap[frame].reshape((len(nonH_atoms), 3)))
            nonH.trajectory.ts.data['dt'] = 1
            nonH.trajectory.ts.data['time'] = frame
            nonH.trajectory.ts.data['step'] = frame
            xtc_writer.write(nonH.atoms)

    return None


# Decoded xyz coordinates to Protein
def extract_template(pdb_file):
    """
        Extract the original PDB template except coordinates
        Input: Template PDB
        Output: template without coordinates
    """
    template = []
    file = open(pdb_file).readlines()
    for line in file:
        if line.startswith("ATOM"):
            template.append(line[:31] + " " * 23 + line[54:])    
        elif line.startswith("TER"):
            template.append(line)
    return template

def load_coor(coordinates, template):
    """
        Create a template with new coordinates
        Input: output xyz coordinates and template without coordinates 
        Output: template with new coordinates
    """
    cur_cor = None
    cur_cor = template[:]
    index = 0
    for i in range(len(coordinates) // 3):
        x, y, z = coordinates[3 * i: 3 * (i + 1)]
        x, y, z = "%.3f" % x, "%.3f" % y, "%.3f" % z
        for cor, end in zip([x, y, z], [38, 46, 54]):
            length = len(cor)
            cur_cor[index] = cur_cor[index][:end - length] \
                + cor + cur_cor[index][end:]

        index += 1
    return cur_cor

def write_file(pdb_file,cur_cor):
    """
        Write the coordinates into PDB
        Inputs: output file name and decoded template
    """
    file = open(pdb_file, "w")
    for line in cur_cor:
        file.write(line)
    file.close()
    return None
