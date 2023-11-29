# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals

# Pre-Training Analysis

def Ramachandran_plot_trj(psf, xtc, out_path):

    u = mda.Universe(psf, xtc)
    protein = u.select_atoms('protein')

    plt.cla()
    plt.clf()
    rama = dihedrals.Ramachandran(protein).run()
    rama.plot(color='black', marker='.', ref=True,s=5)
    plt.savefig(f"{out_path}/ramachandran_org.png")

    return None    

def Ramachandran_plot_decode(phi, psi,out_path, all_hype):

    plt.cla()
    plt.clf()
    ax = plt.gca()
    ax.axis([-180, 180, -180, 180])
    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)
    ax.set(xticks=range(-180, 181, 60), yticks=range(-180, 181, 60),xlabel=r"$\phi$", ylabel=r"$\psi$")
    degree_formatter = plt.matplotlib.ticker.StrMethodFormatter(r"{x:g}$\degree$")
    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)
    ax.scatter(phi, psi, marker='.',s=5)
    plt.savefig(f"{out_path}/{all_hype}_ramachandran_decoded.png")

    return None

def Ramachandran_plot_comp( phi, psi, phi_d, psi_d,out_path, all_hype):

    plt.cla()
    plt.clf()
    ax = plt.gca()
    ax.axis([-180, 180, -180, 180])
    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)
    ax.set(xticks=range(-180, 181, 60), yticks=range(-180, 181, 60),xlabel=r"$\phi$", ylabel=r"$\psi$")
    degree_formatter = plt.matplotlib.ticker.StrMethodFormatter(r"{x:g}$\degree$")
    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)
    ax.scatter(phi, psi, marker='.',s=5,c='b')
    ax.scatter(phi_d, psi_d, marker='.',s=5,c='r',alpha=0.4)
    plt.savefig(f"{out_path}/{all_hype}_ramachandran_comp.png")

    return None


def phi_plot(dihs, orig, all_hype):
    plt.cla()
    plt.clf()
    arr_test = dihs
    ave = arr_test.mean(axis=0)
    std = arr_test.std(axis=0)
    labels = np.arange(2, len(ave)+2, 1)

    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(True)
    ax.plot(labels, ave, lw=2,linestyle="--", c='firebrick')
    ax.errorbar(labels, ave, yerr=std, fmt ='o', lw=2, c='yellow')
    plt.xticks(range(min(labels), max(labels)+1),fontsize=10)
    plt.ylabel(u"$\u03C6$ (in \N{DEGREE SIGN})", fontsize=14)
    plt.xlabel("Residue", fontsize=14)
    fig.subplots_adjust(bottom=0.5)
    plt.savefig(f"{orig}/{all_hype}_phi.png")
    return None


def phi_comp_plot(dihs_org, dihs_decoded,save_path, all_hype):
    plt.cla()
    plt.clf()
    arr_test_org = dihs_org
    ave_org = arr_test_org.mean(axis=0)
    std_org = arr_test_org.std(axis=0)

    arr_test_decoded = dihs_decoded
    ave_decoded = arr_test_decoded.mean(axis=0)
    std_decoded = arr_test_decoded.std(axis=0)

    labels = np.arange(2, len(ave_org)+2, 1)

    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(True)
    ax.plot(labels, ave_org, lw=2,linestyle="--", c='firebrick',label="Original")
    ax.errorbar(labels, ave_org, yerr=std_org, fmt ='o', lw=2, c='gold',label="Original")
    ax.plot(labels, ave_decoded, lw=2,linestyle="--", c='lime', label="Decoded")
    ax.errorbar(labels, ave_decoded, yerr=std_decoded, fmt ='o', lw=2, c='magenta',label="Decoded", alpha=0.5)
    plt.xticks(range(min(labels), max(labels)+1),fontsize=10)
    plt.ylabel(u"$\u03C6$ (in \N{DEGREE SIGN})", fontsize=14)
    plt.xlabel("Residue", fontsize=14)
    fig.subplots_adjust(bottom=0.5)
    plt.legend()
    plt.savefig(f"{save_path}/{all_hype}_comp_phi.png")
    return None

def psi_comp_plot(dihs_org, dihs_decoded,save_path, all_hype):
    plt.cla()
    plt.clf()
    arr_test_org = dihs_org
    ave_org = arr_test_org.mean(axis=0)
    std_org = arr_test_org.std(axis=0)

    arr_test_decoded = dihs_decoded
    ave_decoded = arr_test_decoded.mean(axis=0)
    std_decoded = arr_test_decoded.std(axis=0)

    labels = np.arange(1, len(ave_org)+1, 1)

    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(True)
    ax.plot(labels, ave_org, lw=2,linestyle="--", c='firebrick',label="Original")
    ax.errorbar(labels, ave_org, yerr=std_org, fmt ='o', lw=2, c='gold',label="Original")
    ax.plot(labels, ave_decoded, lw=2,linestyle="--", c='lime', label="Decoded")
    ax.errorbar(labels, ave_decoded, yerr=std_decoded, fmt ='o', lw=2, c='magenta',label="Decoded",alpha=0.5)
    plt.xticks(range(min(labels), max(labels)+1),fontsize=10)
    plt.ylabel(u"$\u03C8$ (in \N{DEGREE SIGN})", fontsize=14)
    plt.xlabel("Residue", fontsize=14)
    fig.subplots_adjust(bottom=0.5)
    plt.legend()
    plt.savefig(f"{save_path}/{all_hype}_comp_psi.png")
    return None


def psi_plot(dihs, orig, all_hype):
    plt.cla()
    plt.clf()    
    arr_test = dihs
    ave = arr_test.mean(axis=0)
    std = arr_test.std(axis=0)
    labels = np.arange(1, len(ave)+1, 1) 

    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(True)
    ax.plot(labels, ave, lw=2,linestyle="--", c='firebrick')
    ax.errorbar(labels, ave, yerr=std, fmt ='o', lw=2, c='darkblue')
    plt.xticks(range(min(labels), max(labels)+1),fontsize=10)
    plt.ylabel(u"$\u03C8$ (in \N{DEGREE SIGN})", fontsize=14)
    plt.xlabel("Residue", fontsize=14)
    fig.subplots_adjust(bottom=0.5)
    plt.savefig(f"{orig}/{all_hype}_psi.png")
    return None

# Latent space

def latent_space_plot(encoded, save_path):
    # Latent_encoded
    a = encoded[0]
    plt.cla()
    plt.clf()
    plt.scatter(a[:,0],a[:,1],c='r')
    plt.savefig(f"{save_path}/encoder_mean.png")

    # Latent_Mean
    b = encoded[1]
    plt.cla()
    plt.clf()
    plt.scatter(b[:,0],b[:,1],c='b',s=5)
    plt.savefig(f"{save_path}/encoder_variance.png")

    # Latent_variance
    c = encoded[2]
    plt.cla()
    plt.clf()
    plt.scatter(c[:,0],c[:,1],c='g',s=5)
    plt.savefig(f"{save_path}/plt_encoded.png")

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
    plt.title(f'loss{all_hype}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{out_path}/{all_hype}_loss.png')
#    plt.show()
    return None
# All loss plot at once


# RMSD, Pearsons, and Spearmann
def Testing_analysis_plot(summary, out_path):
    
    Spearmann_arr = []
    Pearson_arr = []
    RMSD_arr = []
    count = []
    hyper_together_arr = []
    i = 1
    for model in summary:
        Spearmann_arr.append(model['Spearmann'])
        Pearson_arr.append(model['Pearson'])
        RMSD_arr.append(model['RMSD_mean'])
        hyper_together_arr.append(model['hyper_together'])
        count.append(i)
        i +=  1
        

    # Plot Spearmann
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.box(False)
    plt.plot(count,Spearmann_arr )
    plt.scatter(count, Spearmann_arr )
    for i, txt in enumerate(hyper_together_arr):
        plt.annotate(txt, # this is the text
                     (count[i],Spearmann_arr[i]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     ha='center',
                     xytext=(1,-1),
          #           rotation=40,
                     fontsize=10)
    plt.xticks(fontsize=14)
    plt.xlabel('Hyperparameters', fontsize=16)
    plt.ylabel('Spearmann Correlation', fontsize=16)
    plt.savefig(f"{out_path}/Spearmann_arr.png")


    # Plot Pearson
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.box(False)
    plt.plot(count, Pearson_arr)
    plt.scatter(count, Pearson_arr)
    for i, txt in enumerate(hyper_together_arr):
        plt.annotate(txt, # this is the text
                     (count[i],Pearson_arr[i]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     ha='center',
                     xytext=(1,-1),
         #            rotation=80,
                     fontsize=10)
    plt.xticks(fontsize=14)
    plt.xlabel('Hyperparameters', fontsize=16)
    plt.ylabel('Pearson Correlation', fontsize=16)
    plt.savefig(f"{out_path}/Pearson_arr.png")    
    
    # Plot RMSD
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.box(False)
    plt.plot(count, RMSD_arr)
    plt.scatter(count, RMSD_arr)
    for i, txt in enumerate(hyper_together_arr):
        plt.annotate(txt, #this is the text
                     (count[i],RMSD_arr[i]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     ha='center',
                     xytext=(1,-1),
        #             rotation=80,
                     fontsize=10)
    plt.xticks(fontsize=14)
    plt.xlabel('Hyperparameters', fontsize=16)
    plt.ylabel('RMSD', fontsize=16)
    plt.savefig(f"{out_path}/RMSD_arr.png")

    return None
        
    

# Post-training Analysis
def Post_training_analysis_plot(rmsd, c_resids, rmsf, Rgyr, helicity_ave, sheet_ave, out_path, all_hype, names):

    """
    Plot all the analysis on the decoded trajectory
    """    
    if names[2] != 'Protein':
    # RMSD
        plt.cla()
        plt.clf()
        plt.figure(figsize=(20,10))
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        a = 0
        for i, rm in enumerate(rmsd[2:]):
            plt.plot(rmsd[1]/1000,rm, linewidth=1 ,  alpha=1-a, label=names[i+2])
            a = 1/len(rmsd) + a
        plt.ylabel("RMSD (C$\u03B1$) in $\AA$", fontsize=16)
        plt.xlabel('time (ns)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)
        plt.savefig(f"{out_path}/{all_hype}_protein_rmsd.png")

     # RMSD Histogram
        for i, rm in enumerate(rmsd[2:]):
            plt.cla()
            plt.clf()
            plt.figure(figsize=(10,6))
            plt.grid(color='grey', linestyle='--', linewidth=0.5)
            plt.hist(rm, linewidth=1 ,  alpha=0.7, bins=60, color='b', edgecolor='black')
            plt.xlabel(f"RMSD (C$\u03B1$ {names[i+2]}) in $\AA$", fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=16)
            plt.savefig(f"{out_path}/{all_hype}_protein_rmsd_hist_{names[i+2]}.png")

    
    # RMSF
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(c_resids, rmsf)
    plt.scatter(c_resids, rmsf)
    plt.ylabel("RMSF (C$\u03B1$) in $\AA$", fontsize=16)
    plt.xlabel('Residues', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_protein_rmsf.png")

    # Rg
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(rmsd[1]/1000, Rgyr,  linestyle='--', linewidth=1)
    plt.ylabel("$R_g$ ($\AA$)", fontsize=16)
    plt.xlabel('Residues', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_Rg.png")


    # Rg Probability
    # normalization
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    weight=np.ones_like(Rgyr)/float(len(Rgyr))
    x1=np.histogram(Rgyr,bins=np.linspace(2,13,50),weights=weight)
    xdat=(x1[1][0:-1]+x1[1][1:])/2
    y1dat=x1[0]
    plt.plot(xdat,y1dat)
    plt.ylim([0,1])
    plt.xlim([min(Rgyr)-0.5, max(Rgyr)+0.5])
    plt.ylabel('Probability',fontsize=16)
    plt.xlabel('$R_g$ ($\AA$)',fontsize=16)
    plt.tick_params(axis='both', labelsize=14,direction='in',length=8,width=2)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.savefig(f"{out_path}/{all_hype}_Rg_prob.png")

    if names[2] == 'Protein':
        print("HERE")
        c_resids = []
        [c_resids.append(i) for i in range(2,len(helicity_ave)+2,1)]
    # Helicity
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(c_resids, helicity_ave)
    plt.scatter(c_resids, helicity_ave)
    plt.ylabel("Helicilty",fontsize=16)
    plt.xlabel('Residues',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_helicity.png")

    # Sheet 
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(c_resids, sheet_ave)
    plt.scatter(c_resids, sheet_ave)
    plt.ylabel("Sheet",fontsize=16)
    plt.xlabel('Residues',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_Sheet.png")

    return None

def Post_training_analysis_plot_comp(rmsd1, c_resids1, rmsf1, Rgyr1, helicity_ave1, sheet_ave1, out_path, all_hype, rmsd2, c_resids2, rmsf2, Rgyr2, helicity_ave2, sheet_ave2, names):

    """
    Plot all the analysis on the decoded trajectory
    """
    if names[2] != 'Protein':
    # RMSD
        for i, rm in enumerate(rmsd1[2:]):
            plt.cla()
            plt.clf()
            plt.figure(figsize=(20,10))
            plt.grid(color='grey', linestyle='--', linewidth=0.5)
            plt.plot(rmsd2[1]/1000,rm, linewidth=1 , alpha=1,  label = "Original")
            plt.plot(rmsd2[1]/1000,rmsd2[i+2], linewidth=1 , alpha=0.5, label = f"Decoded({all_hype})")
            plt.ylabel(f"RMSD (C$\u03B1$ {names[i+2]}) in $\AA$", fontsize=16)
            plt.xlabel('time (ns)', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=16)
            plt.savefig(f"{out_path}/{all_hype}_protein_rmsd_{names[i+2]}_comp.png")

        # RMSD histogram
        for i, rm in enumerate(rmsd1[2:]):
            plt.cla()
            plt.clf()
            plt.figure(figsize=(10,6))
            plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
            plt.hist(rm, bins=60, color='r', edgecolor='black', alpha=0.7, label = "Original")
            plt.hist(rmsd2[i+2], bins=60, color='b', edgecolor='black', alpha=0.7, label=f"Decoded({all_hype})")
            plt.xlabel(f'RMSD ({names[i+2]} in $\AA$)', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=16)
            plt.savefig(f"{out_path}/{all_hype}_protein_rmsd_hist_{names[i+2]}_comp.png")

    # RMSF
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(c_resids1, rmsf1,'--o', label = "Original")
    plt.plot(c_resids1, rmsf2,'--o', label= f"Decoded({all_hype})")
    plt.ylabel("RMSF (C$\u03B1$) $\AA$", fontsize=16)
    plt.xlabel('Residues', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_protein_rmsf_comp.png")

    # Rg
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(rmsd2[1]/1000, Rgyr1,  linestyle='--', linewidth=1, label = "Original")
    plt.plot(rmsd2[1]/1000, Rgyr2,  linestyle='--', linewidth=1, label = f"Decoded({all_hype})")
    plt.ylabel("$R_g$ ($\AA$)", fontsize=16)
    plt.xlabel('Residues', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_Rg_comp.png")

    # Rg Probability
    # normalization
    plt.cla()
    plt.clf()
    weight1=np.ones_like(Rgyr1)/float(len(Rgyr1))
    x1=np.histogram(Rgyr1,bins=np.linspace(2,13,50),weights=weight1)
    x1dat=(x1[1][0:-1]+x1[1][1:])/2
    y1dat=x1[0]
    weight2=np.ones_like(Rgyr2)/float(len(Rgyr2))
    x2=np.histogram(Rgyr2,bins=np.linspace(2,13,50),weights=weight2)
    xdat2=(x2[1][0:-1]+x2[1][1:])/2
    y2dat=x2[0]
    plt.plot(x1dat,y1dat, label = "Original")
    plt.plot(xdat2,y2dat, label=f"Decoded({all_hype})")
    plt.ylim([0,1])
    plt.xlim([min(Rgyr1)-0.5, max(Rgyr1)+0.5])
    plt.ylabel('Probability',fontsize=16)
    plt.xlabel('$R_g$ ($\AA$)',fontsize=16)
    plt.tick_params(axis='both', labelsize=14,direction='in',length=8,width=2)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.legend(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_Rg_prob_comp.png")

    if names[2] == 'Protein':
        print("HERE")
        c_resids1 = []
        [c_resids1.append(i) for i in range(2,len(helicity_ave1),1)]
    # Helicity
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.grid(color='grey', linestyle='--', linewidth=0.5) 
    plt.plot(c_resids1, helicity_ave1,'--o', label = "Original")
#   plt.scatter(c_resids1, helicity_ave1, label = "Original")
    plt.plot(c_resids1, helicity_ave2, '--o', label=f"Decoded({all_hype})")
#   plt.scatter(c_resids1, helicity_ave2, label="all_hype")
    plt.ylabel("Helicilty",fontsize=16)
    plt.xlabel('Residue Number',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_helicity_comp.png")

    # Sheet 
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.plot(c_resids1, sheet_ave1,'--o',label = "Original")
    plt.plot(c_resids1, sheet_ave2, '--o', label=f"Decoded({all_hype})")
    plt.ylabel("Sheet",fontsize=16)
    plt.xlabel('Residue Number',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"{out_path}/{all_hype}_Sheet_comp.png")

    return None

# plot distribution
def plot_distribution(array_data, bins=100):
    data = array_data.flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(data, bins, density=True, alpha=0.75, color='green', edgecolor='gray')
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # ax.set_ylim([0, 1])
    ax.set_xlabel("CV", fontsize=16)
    ax.set_ylabel("Probability", fontsize=16)
    ax.tick_params(axis='both', which='major', length=10, labelsize=16)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid()
    plt.show()
    return None

# Plot the distributions of all the covalent
def post_training_distribution(covalent, cov_names, bins=100):
    for c, n in zip(covalent, cov_names):
        plt.cla()
        plt.clf()
        data = c.flatten()
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(data, bins, density=True, alpha=0.75, color='green', edgecolor='gray')
        # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        # ax.set_ylim([0, 1])
        ax.set_xlabel("CV", fontsize=16)
        ax.set_ylabel("Probability", fontsize=16)
        ax.tick_params(axis='both', which='major', length=10, labelsize=16)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid()
        plt.savefig(f"CV{n}.png")
        plt.show()


# Plot Gaussian distribution
def plot_Gaussion(x,y):
    data = x.flatten()
    bin_number=100
    
    fig, ax = plt.subplots(figsize=(8, 8))
    n, bins, patches = ax.hist(data, bin_number, density=True, alpha=0.75, color='green', edgecolor='gray')
    ax.plot(x, y/(x[1] - x[0]), 'r--')
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # ax.set_ylim([0, 1])
    ax.set_xlabel("CV", fontsize=16)
    ax.set_ylabel("Probability", fontsize=16)
    ax.tick_params(axis='both', which='major', length=10, labelsize=16)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()
    return None


# Plot the Gaussian distributions of all the covalent
def post_training_Gaussian(y, x, cov_names):
    for g, c, n in zip(y,x, cov_names):
        data = x.flatten()

        bin_number=100
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        n, bins, patches = ax.hist(data, bin_number, density=True, alpha=0.75, color='green', edgecolor='gray')
        ax.plot(x, y/(x[1] - x[0]), 'r--')
        ax.set_xlabel("CV", fontsize=16)
        ax.set_ylabel("Probability", fontsize=16)
        ax.tick_params(axis='both', which='major', length=10, labelsize=16)
        plt.show()
    return None




