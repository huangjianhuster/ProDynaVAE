# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# generate input json file for main.py
# this json will have all arguments
#       keys: trj, psf, datapath, BATCH_SIZE, LATENT_DIM, NUM_HIDDEN_LAYER, EPOCHS, RATE

# Dependencies
import json

input_json = {}
input_json['trj'] = "/home/shrishti/Documents/Projects/IDP_ensemble/ProDynaVAE/ProDynaVAE/ProG_200K/prod_step.xtc"
input_json['psf'] = "/home/shrishti/Documents/Projects/IDP_ensemble/ProDynaVAE/ProDynaVAE/ProG_200K/step3_input.psf"
input_json['pdb'] = "/home/shrishti/Documents/Projects/IDP_ensemble/ProDynaVAE/ProDynaVAE/ProG_200K/step3_input.pdb"
input_json['datapath'] = "ProG_200K/" # better to give an absolute path
input_json['early_stopping'] = True
input_json['seed'] = 4
input_json['post_analysis'] = True #If want to plot and calculate other things
input_json['input_type'] = "cartesian" # Other option is "all"

# the following arguments have to be list type
input_json['BATCH_SIZE'] = [100 ]
input_json['LATENT_DIM'] = [4]
input_json['NUM_HIDDEN_LAYER'] = [4]
input_json['EPOCHS'] = [10000 ]
input_json['RATE'] = [0.0005]

json_object = json.dumps(input_json, indent=4)
with open("../json/sample.json", "w") as outfile:
    outfile.write(json_object)

