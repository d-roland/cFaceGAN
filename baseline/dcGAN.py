# https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

from __future__ import print_function, division

# from keras.datasets import mnist
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D
# from keras.models import Sequential, Model
# from keras.optimizers import Adam

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import VarianceScaling

import matplotlib.pyplot as plt

import sys

import numpy as np

import cv2
from data.celebA import CelebA
import os
import imageio

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        # self.img_rows = 28
        # self.img_cols = 28
        # self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        optimizer2 = optimizer#Adam(.001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer2)

    def build_generator(self):

        model = Sequential()

        # # original
        # model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        # model.add(Reshape((16, 16, 128)))
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=3, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        # model.add(Activation("tanh"))

        # from https://github.com/jazzsaxmafia/dcgan_tensorflow/tree/master/face
        model.add(Dense(1024 * 4 * 4, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Reshape((4, 4, 1024)))
        model.add(Conv2DTranspose(512, kernel_size=5, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(256, kernel_size=5, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(128, kernel_size=5, strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(3, kernel_size=5, strides=(2,2), padding='same'))
        model.add(Activation("tanh"))

        # # from https://github.com/jazzsaxmafia/dcgan_tensorflow/tree/master/face
        # model.add(Dense(1024 * 4 * 4, input_dim=self.latent_dim))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation('relu'))
        # model.add(Reshape((4, 4, 1024)))
        # model.add(UpSampling2D())
        # model.add(Conv2D(512, kernel_size=5, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())
        # model.add(Conv2D(256, kernel_size=5, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=5, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())
        # model.add(Conv2D(3, kernel_size=5, padding="same"))
        # model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # original
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # # from https://github.com/jazzsaxmafia/dcgan_tensorflow/tree/master/face
        # # this one performs quite badly
        # model.add(Conv2D(128, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(1024, kernel_size=5, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Flatten())
        # model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #
        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        face_image_path = "C:/celeba-dataset/img_align_celeba"
        face_images = np.array(list(filter(lambda x: x.endswith('jpg'), os.listdir(face_image_path))))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]
            idx = np.random.randint(0, len(face_images), batch_size)
            imgs = np.array(list(map(lambda x: self.crop_resize( os.path.join( face_image_path, x) ), face_images[idx])))

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            num_generator_cycles=5
            for generator_cycle in range(num_generator_cycles):
                g_loss = self.combined.train_on_batch(noise, valid)
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, imgs)

    def save_imgs(self, epoch, imgs):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                # axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("C:/out/%d.png" % epoch)
        plt.close()
        #
        # imgs_back = 0.5 * imgs + 0.5
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i, j].imshow(imgs_back[cnt, :, :, :])
        #         axs[i, j].axis('off')
        #         cnt += 1
        # fig.savefig("C:/out/actual%d.png" % epoch)
        # plt.close()

    def crop_resize(self, image_path, resize_shape=(64,64)):
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
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=20)