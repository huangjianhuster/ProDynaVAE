# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT


def gen_parms_combinations(**params_dict):
    """
    params_dit: searching grid for each hyperparameter
    it has to contain the following keys:
    BATCH_SIZE: [64,]
    LATENT_DIM: [2, 5, 10]
    NUM_HIDDEN_LAYER: [3, 4, 5]
    EPOCHS: [100, 200, 500]
    RATE: [0.001, 0.01, 0.0001]
    """
    hyperparams_list = [
                  params_dict['BATCH_SIZE'],
                  params_dict['LATENT_DIM'],
                  params_dict['NUM_HIDDEN_LAYER'],
                  params_dict['EPOCHS'],
                  params_dict['RATE'],
                 ]
    combinations = list(itertools.product(*hyperparams_list))
    # reconfigure combinations to dict
    hyperparams_combinations = []
    for i in combinations:
        dict_tmp = {}
        dict_tmp['BATCH_SIZE'], dict_tmp['LATENT_DIM'], dict_tmp['NUM_HIDDEN_LAYER'], dict_tmp['EPOCHS'], dict_tmp['RATE']= i

        hyperparams_combinations.append(i)
    return hyperparams_combinations

def get_ic(psf, xtc):
    u = mda.Universe(psf, dcd)
    selection = u.select_atoms("protein")

    # BAT run
    R = BAT(selection)
    R.run()

    # split EC, IC (bonds, angles and dihedrals)
    Ic = R.results.bat[:,9:]
    Ec = R.results.bat[0,:9].reshape(1,9)

    # split Ic to get bonds, angles and dihedrals
    bonds = R.results.bat[:, 9:len(u.atoms)-3+9]
    angles = R.results.bat[:, len(u.atoms)-3+9:(len(u.atoms)-3)*2+9]
    dihedrals = R.results.bat[:, (len(u.atoms)-3)*2+9:]
    return Ec, bonds, angles, dihedrals

def scaling_spliting(arr):
    """
    arr: the input array
    return scaler, x_test, x_train
    """
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(arr)

    # split dataset into testing and training
    x_test = scale[::4]
    x_train = np.delete(scale, list(range(0, scale.shape[0], 4)), axis=0)

    return scaler, x_test, x_train

