import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import time

# Hyperparameters
latent_dim = 100
epochs = 100000  # This should run for a few hours
batch_size = 64
save_interval = 1000  # Save generated images every 1000 epochs

# Check for GPU availability for performance boost
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Create directories for saved images if not already created
os.makedirs("generated_images", exist_ok=True)

# Generate random noise to initialize generator's training
def generate_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, (batch_size, latent_dim))

# Build the Generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28 * 28, activation="tanh"))
    model.add(Reshape((28, 28)))
    return model

# Build the Discriminator model
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator((28, 28))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False

# Build combined model (generator + discriminator)
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training function
def train(epochs, batch_size, save_interval):
    # Load or generate training data (random patterns)
    X_train = np.random.rand(60000, 28, 28) * 2 - 1  # 60k random 28x28 images

    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]

        noise = generate_noise(half_batch, latent_dim)
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = generate_noise(batch_size, latent_dim)
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        # Print the progress
        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss}]")
            save_generated_images(epoch)

# Save generated images to track training progress
def save_generated_images(epoch, rows=5, cols=5):
    noise = generate_noise(rows * cols, latent_dim)
    gen_imgs = generator.predict(noise)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"generated_images/epoch_{epoch}.png")
    plt.close()

# Start training
start_time = time.time()
with tf.device(device_name):  # Use GPU if available
    train(epochs, batch_size, save_interval)
print(f"Training complete in {(time.time() - start_time) / 3600:.2f} hours.")

