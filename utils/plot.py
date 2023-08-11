# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals

# Pre-Training Analysis

def Ramachandran_plot(psf, xtc, out_path):

    u = mda.Universe(psf, xtc)
    protein = u.select_atoms('protein')

    plt.cla()
    plt.clf()
    rama = dihedrals.Ramachandran(protein).run()
    rama.plot(color='black', marker='.', ref=True,s=5)
    plt.savefig(f"{out_path}/ramachandran.png")

    return None    

def phi_plot(dihs, save_path):

    arr_test = dihs
    ave = arr_test.mean(axis=0)
    std = arr_test.std(axis=0)
    labels = np.arange(2, len(protein.residues)+1, 1)

    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(True)
    ax.plot(labels, ave, lw=2,linestyle="--", c='firebrick')
    ax.errorbar(labels, ave, yerr=std, fmt ='o', lw=2, c='darkblue')
    plt.xticks(range(min(labels), max(labels)+1),fontsize=10)
    plt.ylabel(u"$\u03C6$ (in \N{DEGREE SIGN})", fontsize=14)
    plt.xlabel("Residue", fontsize=14)
    fig.subplots_adjust(bottom=0.5)
    plt.savefig(f"{save_path}/phi.png")
    plt.show()
    return None

def psi_plot(dihs, save_path):
    
    arr_test = dihs
    ave = arr_test.mean(axis=0)
    std = arr_test.std(axis=0)
    labels = np.arange(2, len(protein.residues)+1, 1)

    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(True)
    ax.plot(labels, ave, lw=2,linestyle="--", c='firebrick')
    ax.errorbar(labels, ave, yerr=std, fmt ='o', lw=2, c='darkblue')
    plt.xticks(range(min(labels), max(labels)+1),fontsize=10)
    plt.ylabel(u"$\u03C8$ (in \N{DEGREE SIGN})", fontsize=14)
    plt.xlabel("Residue", fontsize=14)
    fig.subplots_adjust(bottom=0.5)
    plt.savefig(f"{save_path}/psi.png")
    plt.show()
    return None

# Latent space

def latent_space_plot(encoded, save_path):
    # Latent_encoded
    a = encoded[0]
    plt.cla()
    plt.clf()
    plt.scatter(a[:,0],a[:,1],c='r')
    plt.savefig(f"{save_path}/encoder_mean.png")
    plt.show()

    # Latent_Mean
    b = encoded[1]
    plt.cla()
    plt.clf()
    plt.scatter(b[:,0],b[:,1],c='b',s=5)
    plt.savefig(f"{save_path}/encoder_variance.png")
    plt.show()

    # Latent_variance
    c = encoded[2]
    plt.cla()
    plt.clf()
    plt.scatter(c[:,0],c[:,1],c='g',s=5)
    plt.savefig(f"{save_path}/plt_encoded.png")
    plt.show()

    return None

# losses plot

# summarize history for loss
def train_test_loss_plot(history, out_path,all_hype):
    """
    History of the model it will have loss, val_loss and other metric
    Plotting between the loss/val loss and epoch
    """
    
    plt.cla()
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'loss{out_path}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{out_path}/{all_hype}_loss.png')
#    plt.show()
    return None
# All loss plot at once


# RMSD, Pearsons, and Spearmann
def Testing_analysis_plot(summary, out_path):
    
    Spearmann_arr = summary['Spearmann']
    Pearson_arr = summary['Pearson']
    RMSD_arr = summary['RMSD']
    hyper_together_arr = summary['hyper_together']

    # Plot Spearmann
    plt.cla()
    plt.clf()
    plt.plot(hyper_together_arr,Spearmann_arr )
    plt.scatter(hyper_together_arr, Spearmann_arr )
    plt.ylabel('hyperparameters')
    plt.xlabel('Spearmann')
    plt.savefig(f"{out_path}/Spearmann_arr.png")


    # Plot Pearson
    plt.cla()
    plt.clf()
    plt.plot(hyper_together_arr, Pearson_arr)
    plt.scatter(hyper_together_arr, Pearson_arr)
    plt.ylabel('hyperparameters')
    plt.xlabel('Pearson')
    plt.savefig(f"{out_path}/Pearson_arr.png")    
    
    # Plot RMSD
    plt.cla()
    plt.clf()
    plt.plot(hyper_together_arr, RMSD_arr)
    plt.scatter(hyper_together_arr, RMSD_arr)
    plt.ylabel('hyperparameters')
    plt.xlabel('RMSD')
    plt.savefig(f"{out_path}/RMSD_arr.png")

    return None
        
    

# Post-training Analysis
def Post_training_analysis_plot(rmsd, c_resids, rmsf, Rgyr, residues, helicity_ave, sheet_ave, out_path):

    """
    Plot all the analysis on the decoded trajectory
    """    

    # RMSD
    plt.cla()
    plt.clf()
    plt.plot(rmsd[1],rmsd[2])
    plt.ylabel("RMSD (C$\u03B1$) in $\AA$")
    plt.xlabel('time (ns)')
    plt.savefig(f"{out_path}/protein_rmsd.png")
    
    # RMSF
    plt.cla()
    plt.clf()
    plt.plot(c_resids, rmsf)
    plt.ylabel("RMSF (C$\u03B1$) in $\AA$")
    plt.xlabel('residue')
    plt.savefig(f"{out_path}/residue.png")

    # Rg
    plt.cla()
    plt.clf()
    plt.plot(Rgyr[:,0], Rgyr[:,1])
    plt.ylabel(r"radius of gyration $R_G$ ($\AA$)")
    plt.xlabel("time (ps)")
    plt.savefig(f"{out_path}/Rg.png")

    # Helicity
    plt.cla()
    plt.clf()
    plt.plot(residues, helicity_ave)
    plt.ylabel("Helicilty")
    plt.xlabel('residues')
    plt.savefig(f"{out_path}/helicity.png")

    # Sheet 
    plt.cla()
    plt.clf()
    plt.plot(residues, sheet_ave)
    plt.ylabel("Sheet")
    plt.xlabel('residues')
    plt.savefig(f"{out_path}/Sheet.png")

    return None
