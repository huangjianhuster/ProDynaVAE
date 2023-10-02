# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# generate input json file for main.py
# this json will have all arguments
#       keys: trj, psf, datapath, BATCH_SIZE, LATENT_DIM, NUM_HIDDEN_LAYER, EPOCHS, RATE

# Dependencies
import json

input_json = {}
input_json['trj'] = "/home/shrishti/Documents/Projects/IDP_ensemble/protein-VAE-main_org/charmm-gui_3GB1/gromacs/prod_step_5_align.xtc" 
input_json['psf'] = "/home/shrishti/Documents/Projects/IDP_ensemble/protein-VAE-main_org/charmm-gui_3GB1/step1_pdbreader.psf"
input_json['pdb'] =  "/home/shrishti/Documents/Projects/IDP_ensemble/protein-VAE-main_org/charmm-gui_3GB1/gromacs/prod_step_5_protein.pdb"
input_json['datapath'] = "ProG_20K/" # better to give an absolute path
input_json['early_stopping'] = True
input_json['seed'] = 4
input_json['post_analysis'] = True #If want to plot and calculate other things
input_json['input_type'] = "cartesian" # Other option is "all"
input_json['timestep'] = 5
input_json['selection'] = "protein and backbone and resid 24:37"
input_json['rmsd_names'] = ["Frame", "Time", "Helix", "Backbone", "Resid 8 to 14"]
input_json['rmsd_cal'] = ["protein and backbone", "protein and backbone and resid 8:14"]
input_json['split'] = 0.7

# the following arguments have to be list type
input_json['BATCH_SIZE'] = [100 ]
input_json['LATENT_DIM'] = [4,6,8]
input_json['NUM_HIDDEN_LAYER'] = [4,6,8]
input_json['EPOCHS'] = [10000 ]
input_json['RATE'] = [0.0005]

json_object = json.dumps(input_json, indent=4)
with open("../json/sample.json", "w") as outfile:
    outfile.write(json_object)

