"""
VAE Model Architectures

"""

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
import numpy as np

def Hidden_layer_neurons(latent_dim, original_dim, number_of_hidden_layers):
    # Get number of neurons for each hidden layer

    print(original_dim,"original_dim,")   
    neuron_layer = np.zeros(number_of_hidden_layers + 1).astype(int)
    neuron_layer[0] = latent_dim
    neuron_layer[-1] = original_dim

    max_nodes = original_dim
    min_nodes = latent_dim*5
    neuron_layer[1] = min_nodes
    neuron_layer_betw = []
    for i in range(2, number_of_hidden_layers, 1):
        neuron_layer_betw.append((max_nodes - min_nodes) / 2)
        min_nodes = min(neuron_layer_betw)
        if len(neuron_layer_betw) >= 2:
            max_nodes = max(neuron_layer_betw)
    neuron_layer_betw.sort()
    for i in range(0, len(neuron_layer_betw), 1):
        neuron_layer[i+2] = neuron_layer_betw[i]
    return neuron_layer



def vae_encoder(original_dim, latent_dim=2, num_of_hidden_layer=4):

    """Build VAE encoder model.

    Parameters
    ----------
    original_dim : int

        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    encoder : keras.Model
        constructed encoder model
    z_mean : keras.Model.Dense
    z_log_var : keras.Model.Dense
    encoder_input : keras.Model.Input
        input layer
    """

    input_shape = (original_dim, )
    encoder_input = x = Input(shape=input_shape)

    neuron_layer = Hidden_layer_neurons(latent_dim, original_dim, num_of_hidden_layer)

    for i in range(len(neuron_layer) - 2, 0, -1):
        x = Dense(neuron_layer[i], activation='relu')(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args

        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.
            )

        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    #This Lambda layerfor mean and variance

    # create encoder
    encoder = Model(encoder_input, [z_mean, z_log_var, z])
    print(encoder,"encoder,")
    print(z_mean,"z_mean,")
    print(z_log_var,"z_log_var,")
    print(encoder_input, "encoder_input")
    return encoder, z_mean, z_log_var, encoder_input

def vae_decoder(original_dim, latent_dim=2, num_of_hidden_layer=4):    
    """Build VAE/AE decoder model.
    Can be used for both VAE and AE.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    decoder : keras.Model
        constructed decoder model
    """

    latent_inputs = x = Input(shape=(latent_dim,))

    neuron_layer = Hidden_layer_neurons(latent_dim, original_dim, num_of_hidden_layer)

    for i in range(1, len(neuron_layer) - 1):
        x = Dense(neuron_layer[i], activation='relu')(x)

    x = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, x)
    return decoder



def build_vae(original_dim, latent_dim=2, num_of_hidden_layer=4, rate=0.0001):
    """Build VAE model.
    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    encoder : keras.Model
        constructed encoder model
    decoder : keras.Model
        constructed decoder model
    vae: keras.Model
        constructed VAE model
    """

    # init encoder and decoder
    encoder, z_mean, z_log_var, encoder_input = vae_encoder(
        original_dim, latent_dim, num_of_hidden_layer
        )

    decoder = vae_decoder(original_dim, latent_dim, num_of_hidden_layer)

    # same as: z_decoded = decoder(z)
    z_decoded = decoder(encoder(encoder_input)[2])

    def vae_loss(x, z_decoded):
        # reconstruction loss
        x = K.flatten(x)

        z_decoded = K.flatten(z_decoded)
        xent_loss = losses.binary_crossentropy(x, z_decoded)

        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1
            )

        return K.mean(xent_loss + kl_loss)

    # Instantiate the VAE model:
    vae = Model(encoder_input, z_decoded)
    vae.add_loss(vae_loss(encoder_input, z_decoded))

    opt = Adam(learning_rate=rate)
    vae.compile(optimizer=opt)
    #for layer in vae.layers: print(layer, "layer.get_weights", layer.get_weights())
    return encoder, decoder, vae
