# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import matplotlib.pyplot as plt


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.plot('')
    plt.show()



