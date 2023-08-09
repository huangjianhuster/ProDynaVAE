# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# generate input json file for main.py
# this json will have all arguments
#       keys: trj, psf, datapath, BATCH_SIZE, LATENT_DIM, NUM_HIDDEN_LAYER, EPOCHS, RATE

# Dependencies
import json

input_json = {}
input_json['trj'] = "pro_align.xtc"
input_json['psf'] = "step3_input.psf"
input_json['datapath'] = "../tmp/"
# the following arguments have to be list type
input_json['BATCH_SIZE'] = [100, ]
input_json['LATENT_DIM'] = [2, ]
input_json['NUM_HIDDEN_LAYER'] = [4, ]
input_json['EPOCHS'] = [200, ]
input_json['RATE'] = [0.0001,]

json_object = json.dumps(input_json, indent=4)
with open("../json/sample.json", "w") as outfile:
    outfile.write(json_object)

