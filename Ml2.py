import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.datasets import fetch_openml
from tensorflow.keras.callbacks import Callback
import time

# Load MNIST dataset (already available if you've used it before)
print("Loading MNIST dataset...")
start_time = time.time()
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0  # Normalize to [0,1]
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

# Reshape to add channel dimension
X = X.values.reshape(-1, 28, 28, 1)

# Set training parameters
input_shape = (28, 28, 1)
latent_dim = 2  # Dimension of the latent space
epochs = 200    # Set a high number for prolonged training
batch_size = 64

# Custom callback for logging progress
class ProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {logs['loss']:.4f}")

# VAE encoder
def build_encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    model = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return model

# VAE decoder
def build_decoder(latent_dim, original_shape):
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(256, activation='relu')(latent_inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(np.prod(original_shape), activation='sigmoid')(x)
    outputs = Reshape(original_shape)(x)
    model = Model(latent_inputs, outputs, name="decoder")
    return model

# Loss function
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = MeanSquaredError()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.mean_squared_error(inputs, reconstructed), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.add_loss(reconstruction_loss + kl_loss)
        return reconstructed

# Compile the VAE model
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')

# Train the model with logging every 10 epochs
vae.fit(X, X,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ProgressCallback()],
        verbose=0)  # Set verbose=0 to rely on the callback for output

