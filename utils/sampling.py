import tensorflow as tf
import sys

def sample_distribution(args):
    n_sample, latent_dim, decoder = args
    z_sample = tf.keras.backend.random_normal(shape=(n_sample, latent_dim))
    return decoder.predict(z_sample)


def model_load(path_vae, pah_encoder, path_decoder):
    loaded_vae = tf.keras.models.load_model(path_vae)
    loaded_decoder = tf.keras.models.load_model(path_decoder)
    loaded_encoder = tf.keras.models.load_model(path_encoder)
    return loaded_vae, loaded_decoder, loaded_encoder

latent_dim = sys.argv[0]   # Replace with the actual latent dimension of your model
n_samples =  sys.argv[1]  # Number of samples to generate

# Specify paths to your saved models
path_to_vae_model = sys.argv[2]     # 'path to your vae_model.h5'
path_to_encoder_model = sys.argv[3] # 'path to your encoder_model.h5'
path_to_decoder_model = sys.argv[4] # 'path to your decoder_model.h5'

# Load models
loaded_vae, loaded_decoder, loaded_encoder = model_load(path_to_vae_model, path_to_encoder_model, path_to_decoder_model)

# Sample from latent space
sampled_data = sample_distribution((n_samples, latent_dim, loaded_decoder))

# Display or use the sampled data as needed
print(sampled_data)


