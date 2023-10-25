# ProDynaVAE
a VAE model for protein dynamics

# Dependencies:
# Name                    Version       
_libgcc_mutex             0.1          
biopython                 1.81          
keras-preprocessing       1.1.2         
matplotlib                3.5.3         
mdanalysis                2.1.0         
mdtraj                    1.9.5         
numpy                     1.19.5        
pandas                    1.3.5         
python                    3.7.12      
scikit-learn              0.23.2        
scipy                     1.7.3         
tensorflow                2.4.1      

# Other Dependencies:   
ModRefiner


# Usage

# Making json input file:
trj: Provide path for trajectory file of interested protein
psf: Provide path for psf file of interested protein
pdb: Provide path for pdb file of interested protein
datapath: Path of the folder where you want to store the results
early_stopping: True --> If you want the training to stop if the given min_delta( Minimum change in the monitored quantity to qualify as an improvement) in the validation loss is not changing consequetly given number of epoach (patience). FYI (Can provide more values like start_from_epoch,  restore_best_weights, baseline). False --> The run till the last epoch.
seed: Give random seed in order to reproduce results across different runs.
post_analysis: True --> Do all the analysis for training and decoded results sunch as rmsd calculation, rg, PCA, end-to-end distance, etc. False --> Skip the post analysis.
input_type: such as "cartesian", "calpha", "dihedrals_all", "dihedral_backbone", "contact_map".
timestep: The timestep used in your simulation.
selection: In the protein give the most stable region for the RMSD calculation.
rmsd_names: ["Frame", "Time", Selection, ...]
rmsd_cal: Calculate RMSD of other regions as well.
Below parameteres could be more than one if want to do hyperparameter tuning
BATCH_SIZE: minibatch for the training
LATENT_DIM: dimension of latent sapce
NUM_HIDDEN_LAYER: Number of hidden layers (The code will automatically calculate the number of nodes in each layer)
EPOCHS: epoch
RATE: learning rate

# To run json file
Go to scripts directory
cd scripts
Run
python make_json.py

This will generate the json file for given protein in the json folder.

# To run the main.py file
python main.py --input json/filename


