# Dependencies
import pickle
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import MDAnalysis as mda
from tensorflow.keras.callbacks import Callback
import sys

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
            "EPOCHS": 200
            "RATE": 0.00001
        }

    """
    scaler = kwargs['scaler']
    x_test = kwargs['x_test']
    x_train = kwargs['x_train']
    original_dim = x_train.shape[1]

    BATCH_SIZE = kwargs['BATCH_SIZE']
    LATENT_DIM = kwargs['LATENT_DIM']
    NUM_HIDDEN_LAYER = kwargs['NUM_HIDDEN_LAYER']
    EPOCHS = kwargs['EPOCHS']
    RATE = kwargs['RATE']

    # Build VAE MODEL
    encoder, decoder, vae = build_vae(original_dim, LATENT_DIM, NUM_HIDDEN_LAYER, RATE)
    encoder.summary()

    # Train VAE MODEL
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    vae.fit(x=x_train, y=x_train,
            shuffle=True,
            epochs=EPOCHS,
            validation_data=(x_test, x_test),
            callbacks=[early_stopping])

    # TEST 
    encoded = encoder.predict(x_test, batch_size=BATCH_SIZE)
    decoded = decoder.predict(encoded[0])
    demap =  scaler.inverse_transform(decoded)     
    # Convert back
    sin_demap = demap[:,:int(len(demap[0])/2)]
    cos_demap = demap[:,int(len(demap[0])/2):]

    demap_rad = np.arctan2(sin_demap, cos_demap)

    # Plot here
    #latent_space_plot(encoded, save_path)

    # Evaluation 
    Spearmann, Pearson, RMSD = evaluate(encoded, demap, x_test, scaler) 
    print(Spearmann, Pearson, RMSD)
    # Create return dict
    return_dict = {}
    
    return_dict['BATCH_SIZE'] = BATCH_SIZE
    return_dict['LATENT_DIM'] = LATENT_DIM
    return_dict['NUM_HIDDEN_LAYER'] = NUM_HIDDEN_LAYER
    return_dict['EPOCHS'] = EPOCHS
    return_dict['RATE'] = RATE

    return_dict['Spearmann'] = Spearmann
    return_dict['Pearson'] = Pearson 
    return_dict['RMSD'] = RMSD
 
    return_dict['demap'] = demap_rad

    return return_dict

# Sampling from Latent space
#def sample_latent():



def evaluate(encoded, demap, x_test, scaler):
    Spearmann, Pearson = cal_spearman(x_test, encoded[0])
    RMSD = cal_rmsd(x_test, demap, encoded[0], scaler)
    return Spearmann, Pearson, RMSD

def demap_to_PDB(Ic_bonds, Ic_angles, Ec, torsion,pdb_temp, R):
    Ic_bonds_angles_ave = np.average(np.concatenate([Ic_bonds, Ic_angles], axis=1), axis=0)
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
        write_file(f"{i}.pdb",coorinates)
        file.append(f"{i}.pdb")
        i += 1  
    with open("pickle_file.pkl", 'wb') as w:
        pickle.dump(file, w)
    return None

def PDB_to_XTC(pickle_file, template_file):
    pdb_file = pickle.load(open(pickle_file,'rb'))
    reformat_pdbfiles = [(i) for i in pdb_file]
    u = mda.Universe(template_file)
    with mda.Writer(output_xtc, len(u.atoms)) as xtc_writer:
        for pdb_file1 in reformat_pdbfiles[0:]:
                u.load_new(pdb_file1)  # Load each PDB file into the Universe
                xtc_writer.write(u)
        for temp_file in pdb_file:
            os.system("rm %s" % temp_file)
    return None

# Spearmann correlation and Pearson correlation
def cal_spearman(data_original, data_encoded):
    dist_encoded = np.square(euclidean_distances(data_encoded, data_encoded)).flatten()
    dist_original = np.square(euclidean_distances(data_original, data_original)).flatten()
    spearman = spearmanr(dist_original, dist_encoded)
    pearson = pearsonr(dist_original, dist_encoded)
    return spearman, pearson

# RMSD in A
def cal_rmsd(data_original, data_decoded, data_encoded, scaler):
    data_decoded = data_decoded
    #data_decoded = data_decoded.reshape(data_encoded.shape[0], -1)    
    a = np.sum(np.square(data_decoded*10 - data_original*10), axis=1)
    rmsd = np.sqrt(a / (data_original.shape[1] // 3))
    return np.mean(rmsd), np.std(rmsd)

# Decoded to Protein
def extract_template(pdb_file):
    template = []
    file = open(pdb_file).readlines()
    for line in file:
        if line.startswith("ATOM"):
            template.append(line[:31] + " " * 23 + line[54:])    
        elif line.startswith("TER"):
            template.append(line)
    return template

def load_coor(coordinates, template):
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
    file = open(pdb_file, "w")
    for line in cur_cor:
        file.write(line)
    file.close()
    return None
