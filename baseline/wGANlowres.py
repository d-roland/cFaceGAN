# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py

from __future__ import print_function, division

# from keras.datasets import mnist
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D
# from keras.models import Sequential, Model
# from keras.optimizers import RMSprop

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np
import os
import imageio
import cv2

class WGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        # original
        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        # # from https://github.com/jazzsaxmafia/dcgan_tensorflow/tree/master/face
        # # not good
        # model.add(Dense(1024 * 4 * 4, input_dim=self.latent_dim))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation('relu'))
        # model.add(Reshape((4, 4, 1024)))
        # model.add(Conv2DTranspose(512, kernel_size=5, strides=(2,2), padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(Conv2DTranspose(256, kernel_size=5, strides=(2,2), padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(Conv2DTranspose(128, kernel_size=5, strides=(2,2), padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(Conv2DTranspose(3, kernel_size=5, strides=(2,2), padding='same'))
        # model.add(Activation("tanh"))

        # # FC model
        # model.add(Dense(256, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(4096))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #
        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        face_image_path = "C:/celeba-dataset/img_align_celeba"
        face_images = np.array(list(filter(lambda x: x.endswith('jpg'), os.listdir(face_image_path))))

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                # imgs = X_train[idx]
                idx = np.random.randint(0, len(face_images), batch_size)
                imgs = np.array(
                    list(map(lambda x: self.crop_resize(os.path.join(face_image_path, x)), face_images[idx])))

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("C:/out/%d.png" % epoch)
        plt.close()

    def crop_resize(self, image_path, resize_shape=(32,32)):
        image = imageio.imread(image_path)
        height, width, channel = image.shape

        if width == height:
            resized_image = cv2.resize(image, resize_shape)
        elif width > height:
            resized_image = cv2.resize(image, (int(width * float(resize_shape[0])/height), resize_shape[1]))
            cropping_length = int( (resized_image.shape[1] - resize_shape[0]) / 2)
            resized_image = resized_image[:,cropping_length:cropping_length+resize_shape[1]]
        else:
            resized_image = cv2.resize(image, (resize_shape[0], int(height * float(resize_shape[1])/width)))
            cropping_length = int( (resized_image.shape[0] - resize_shape[1]) / 2)
            resized_image = resized_image[cropping_length:cropping_length+resize_shape[0], :]

        return resized_image/127.5 - 1

if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)