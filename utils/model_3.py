"""
VAE Model Architectures

"""
#import os

#os.environ["keras.backend.BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
#import keras
#from keras.backend.import layers

def Hidden_layer_neurons(latent_dim, original_dim, number_of_hidden_layers):
    # Get number of neurons for each hidden layer
    neuron_layer = np.zeros(number_of_hidden_layers + 2).astype(int)
    neuron_layer[0] = latent_dim
    neuron_layer[-1] = original_dim

    max_nodes = original_dim
    min_nodes = latent_dim*5
    neuron_layer[1] = min_nodes
    neuron_layer_betw = []
    for i in range(2, number_of_hidden_layers+1, 1):
        neuron_layer_betw.append((max_nodes - min_nodes) / 2)
        min_nodes = min(neuron_layer_betw)
        if len(neuron_layer_betw) >= 2:
            max_nodes = max(neuron_layer_betw)
    neuron_layer_betw.sort()
    for i in range(0, len(neuron_layer_betw), 1):
        neuron_layer[i+2] = neuron_layer_betw[i]
    return neuron_layer

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.keras.backend.shape(z_mean)[0]
        dim = tf.keras.backend.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def enc_dec(original_dim, latent_dim=2, num_of_hidden_layer=4):

    input_shape = (original_dim, )
    encoder_input = x = tf.keras.layers.Input(shape=input_shape)

    neuron_layer = Hidden_layer_neurons(latent_dim, original_dim, num_of_hidden_layer)

    for i in range(len(neuron_layer) - 2, 0, -1):
        x = tf.keras.layers.Dense(neuron_layer[i], activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = x = tf.keras.layers.Input(shape=(latent_dim,))

    neuron_layer = Hidden_layer_neurons(latent_dim, original_dim, num_of_hidden_layer)

    for i in range(1, len(neuron_layer) - 1):
        x = tf.keras.layers.Dense(neuron_layer[i], activation='relu')(x)

    x = tf.keras.layers.Dense(original_dim, activation='sigmoid')(x)

    decoder = tf.keras.Model(latent_inputs, x, name="decoder")
    decoder.summary()
    return encoder, decoder

def metrics(total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker):
    return [total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker]

#def train_step(data, encoder, decoder, optimizer, trainable_weights,
#              total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker):
#    with tf.GradientTape() as tape:
#        z_mean, z_log_var, z = encoder(data)
#        reconstruction = decoder(z)
#        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
#            tf.keras.backend.losses.binary_crossentropy(data, reconstruction),
#            axis=(-1,)
#        ))
#        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#        total_loss = reconstruction_loss + kl_loss
#
#    grads = tape.gradient(total_loss, trainable_weights)
#    optimizer.apply_gradients(zip(grads, trainable_weights))
#    total_loss_tracker.update_state(total_loss)
#    reconstruction_loss_tracker.update_state(reconstruction_loss)
#    kl_loss_tracker.update_state(kl_loss)
#
#    return {
#        "loss": total_loss_tracker.result(),
#        "reconstruction_loss": reconstruction_loss_tracker.result(),
#        "kl_loss": kl_loss_tracker.result(),
#    }

class VAE(tf.keras.models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        """Call the model on a particular input."""
        z_mean, z_log_var, z = encoder(inputs)
        reconstruction = decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        """Step run during training."""
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                BETA
                * losses.binary_crossentropy(
                    data, reconstruction, axis=-1
                )
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5
                    * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Step run during validation."""
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            BETA
            * losses.binary_crossentropy(data, reconstruction, axis=-1)
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

#def train_step(data, encoder, decoder, optimizer, trainable_weights,
#               total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker):
#    with tf.GradientTape() as tape:
#        z_mean, z_log_var, z = encoder(data)
#        reconstruction = decoder(z)
#        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
#            tf.keras.losses.binary_crossentropy(data, reconstruction),
#            axis=-1
#        ))
#        kl_loss = -0.5 * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var))
#        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#        total_loss = tf.keras.backend.sum(reconstruction_loss + kl_loss)
#    grads = tape.gradient(total_loss, trainable_weights)
#    optimizer.apply_gradients(zip(grads, trainable_weights))
#
#    total_loss_tracker.update_state(total_loss)
#    reconstruction_loss_tracker.update_state(reconstruction_loss)
#    kl_loss_tracker.update_state(kl_loss)
#
#    return {
#        "loss": total_loss_tracker.result(),
#        "reconstruction_loss": reconstruction_loss_tracker.result(),
#        "kl_loss": kl_loss_tracker.result(),
#    }
#
#
#def VAE(encoder, decoder):
#    total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#    reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
#    kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
#
#    return total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker
#
def build_vae(original_dim, latent_dim=2, num_of_hidden_layer=4, rate=0.0001):
    encoder, decoder = enc_dec(original_dim, latent_dim, num_of_hidden_layer)
    vae = VAE(encoder, decoder)
#    input_shape = (original_dim, ) 
#    # Get trainable weights and create optimizer
#    trainable_weights = encoder.trainable_weights + decoder.trainable_weights
#    optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
#
#    # Initialize metrics trackers
#    total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker = VAE(encoder, decoder)
#
#    # Perform a single training step
#    results = train_step(input_shape, encoder, decoder, optimizer, trainable_weights,
#                         total_loss_tracker, reconstruction_loss_tracker, kl_loss_tracker)
#    
#    # Uncomment the lines below if you want to create a complete VAE model and compile it
#    vae = tf.keras.backend.Model(input_shape, outputs=decoder(encoder(input_shape)))
#    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=rate))
#
#    return encoder, decoder, vae, results

