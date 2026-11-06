# -*- coding: utf-8 -*-
"""Minimal repair of the supplied Colab colorization GAN.

The original CIFAR-10 images remain 32x32 throughout.  Upscaling CIFAR-10 to
128x128 would not add visual information, so the networks and previews stay at
the native resolution.

Run this file in a Colab notebook with ``%run image_colorization_real_images_minimal.py``
so the epoch preview figures are rendered inline.  Running it with ``!python``
still saves the previews but cannot display matplotlib figures in the notebook.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display

from google.colab import drive
from sklearn.utils import shuffle
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout
from keras.layers import BatchNormalization, LeakyReLU, Flatten
from keras.layers import Conv2DTranspose, Concatenate, Dot
from keras.optimizers import Adam
import tensorflow as tf


IMAGE_SIZE = 32
DATA_SIZE = 32
OUTPUT_DIR = Path("colorization_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USE_GOOGLE_DRIVE = False  # Set True only if you want weights persisted in Drive.


def plot(gs, colorized, original, n):
    """Show grayscale input, generated RGB, and original RGB in three rows."""
    samples = [gs, colorized, original]
    fig = plt.figure(figsize=(2.2 * n, 6))
    grid_spec = gridspec.GridSpec(3, n)
    g = 0

    for i in range(3):
        for j in range(n):
            ax = plt.subplot(grid_spec[g])
            g += 1
            plt.axis("off")
            ax.set_aspect("equal")
            image = samples[i][j]
            if image.shape[-1] == 1:
                plt.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
            else:
                # The networks use [-1, 1], while imshow expects [0, 1].
                plt.imshow(np.clip((image + 1) / 2, 0, 1))
    fig.tight_layout()
    return fig


def resize_batch(images):
    """Keep CIFAR-10 at its native resolution."""
    return images.astype(np.float32, copy=False)


def Generator():
    X_gs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    C1 = LeakyReLU(0.2)(Conv2D(64, kernel_size=1, strides=1)(X_gs))
    C2 = LeakyReLU(0.2)(Conv2D(128, kernel_size=2, strides=2)(C1))
    C3 = LeakyReLU(0.2)(Conv2D(256, kernel_size=2, strides=2)(C2))
    C4 = LeakyReLU(0.2)(Conv2D(512, kernel_size=2, strides=2)(C3))
    C5 = LeakyReLU(0.2)(Conv2D(512, kernel_size=2, strides=2)(C4))

    DC0 = Dropout(0.5)(BatchNormalization()(LeakyReLU(0.2)(Conv2DTranspose(1024, 2, 2)(C5))))
    DC0 = Concatenate(axis=3)([DC0, C4])
    DC1 = Dropout(0.5)(BatchNormalization()(LeakyReLU(0.2)(Conv2DTranspose(512, 2, 2)(DC0))))
    DC1 = Concatenate(axis=3)([DC1, C3])
    DC2 = BatchNormalization()(LeakyReLU(0.2)(Conv2DTranspose(256, 2, 2)(DC1)))
    DC2 = Concatenate(axis=3)([DC2, C2])
    DC3 = BatchNormalization()(LeakyReLU(0.2)(Conv2DTranspose(128, 2, 2)(DC2)))
    DC3 = Concatenate(axis=3)([DC3, C1])
    output = Conv2D(3, 1, 1, activation="tanh")(DC3)
    return Model(X_gs, output, name="generator")


def Discriminator():
    X_gs_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    X_rgb_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    input_images = Concatenate(axis=3)([X_gs_input, X_rgb_input])

    C1 = LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, 2, 2)(input_images)))
    C2 = LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, 2, 2)(C1)))
    C3 = LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, 2, 2)(C2)))
    C4 = LeakyReLU(0.2)(BatchNormalization()(Conv2D(256, 1, 1)(C3)))
    D = Dense(128)(Flatten()(C4))
    D = Dense(1, activation="sigmoid")(D)
    return Model([X_gs_input, X_rgb_input], D, name="discriminator")


def main():
    if USE_GOOGLE_DRIVE:
        drive.mount("/content/gdrive", force_remount=True)
    (x_train, _), (x_test, _) = cifar10.load_data()

    # Keep the dataset compact and at its native 32x32 resolution.
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    X_gs = np.mean(x_train, axis=3, keepdims=True).astype(np.float32)
    X_rgb = (x_train * 2.0 - 1.0).astype(np.float32)
    X_test_gs = np.mean(x_test, axis=3, keepdims=True).astype(np.float32)
    X_test_rgb = (x_test * 2.0 - 1.0).astype(np.float32)
    del x_train, x_test

    generator = Generator()
    discriminator = Discriminator()
    X_gs_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    X_rgb_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    colorized = generator(X_gs_input)
    adversarial_output = discriminator([X_gs_input, colorized])
    cosine_output = Dot(axes=1, normalize=True)([Flatten()(colorized), Flatten()(X_rgb_input)])
    combined = Model([X_gs_input, X_rgb_input], [adversarial_output, cosine_output])

    generator_loss = []
    discriminator_loss = []
    epochs = 25
    batch_size = 64
    n_example = len(X_gs)
    batches = n_example // batch_size
    discriminator_updates = 2
    generator_updates = 1
    zero = np.zeros((batch_size, 1), dtype=np.float32)
    one = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    d_loss_factor = batches * 2 * discriminator_updates
    g_loss_factor = batches * generator_updates

    # Start fresh unless both compatible files already exist.
    if USE_GOOGLE_DRIVE:
        weights_dir = Path("/content/gdrive/My Drive/Image Colorization")
    else:
        weights_dir = OUTPUT_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    generator_path = weights_dir / "Generator.weights.h5"
    discriminator_path = weights_dir / "Discriminator.weights.h5"
    reuse = generator_path.exists() and discriminator_path.exists()
    if reuse:
        generator.load_weights(str(generator_path))
        discriminator.load_weights(str(discriminator_path))
        print("Loaded existing weights.")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)
        g_loss = 0.0
        d_loss = 0.0
        indices = shuffle(np.arange(n_example), random_state=epoch)

        discriminator.trainable = True
        discriminator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001))
        for _ in range(discriminator_updates):
            for b in range(batches):
                batch_ids = indices[b * batch_size:(b + 1) * batch_size]
                x_batch = resize_batch(X_gs[batch_ids])
                y_batch = resize_batch(X_rgb[batch_ids])
                pre_batch = generator.predict(x_batch, verbose=0)
                d_loss += float(discriminator.train_on_batch([x_batch, y_batch], one))
                d_loss += float(discriminator.train_on_batch([x_batch, pre_batch], zero))

        discriminator.trainable = False
        combined.compile(
            loss=["binary_crossentropy", "mse"],
            loss_weights=[1.0, 10.0],
            optimizer=Adam(learning_rate=0.0001),
        )
        for _ in range(generator_updates):
            for b in range(batches):
                batch_ids = indices[b * batch_size:(b + 1) * batch_size]
                x_batch = resize_batch(X_gs[batch_ids])
                y_batch = resize_batch(X_rgb[batch_ids])
                values = combined.train_on_batch([x_batch, y_batch], [one, np.ones((len(batch_ids), 1), dtype=np.float32)])
                g_loss += float(values[0] if isinstance(values, (list, tuple, np.ndarray)) else values)

        g_loss /= g_loss_factor
        d_loss /= d_loss_factor
        generator_loss.append(g_loss)
        discriminator_loss.append(d_loss)
        print(f"Discriminator Loss: {d_loss:.4f}", flush=True)
        print(f"Generator Loss: {g_loss:.4f}", flush=True)

        # Save the epoch preview and display it immediately in a notebook.
        preview_indices = np.arange(8)
        preview_gs = resize_batch(X_test_gs[preview_indices])
        preview_rgb = resize_batch(X_test_rgb[preview_indices])
        preview_color = generator.predict(preview_gs, verbose=0)
        figure = plot(preview_gs, preview_color, preview_rgb, len(preview_indices))
        figure.savefig(OUTPUT_DIR / f"epoch_{epoch + 1:03d}.png", dpi=120)
        display(figure)
        plt.close(figure)

        generator.save_weights(str(generator_path))
        discriminator.save_weights(str(discriminator_path))

    plt.figure(figsize=(8, 4))
    plt.plot(generator_loss, c="r", label="Generator Loss")
    plt.plot(discriminator_loss, c="b", label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
