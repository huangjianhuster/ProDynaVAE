# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import matplotlib.pyplot as plt

def latent_space_plot(encoded, save_path):
    # Latent_encoded
    a = encoded[0]
    plt.cla()
    plt.clf()
    plt.scatter(a[:,0],a[:,1],c='r')
    plt.savefig(f"{save_path}encoder_mean.png")
    plt.show()

    # Latent_Mean
    b = encoded[1]
    plt.cla()
    plt.clf()
    plt.scatter(b[:,0],b[:,1],c='b',s=5)
    plt.savefig(f"{save_path}encoder_variance.png")
    plt.show()

    # Latent_variance
    c = encoded[2]
    plt.cla()
    plt.clf()
    plt.scatter(c[:,0],c[:,1],c='g',s=5)
    plt.savefig(f"{save_path}plt_encoded.png")
    plt.show()

    return None


