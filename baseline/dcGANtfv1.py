# https://github.com/jazzsaxmafia/dcgan_tensorflow/tree/master/face

# import tensorflow as tf
import os
import pandas as pd
import numpy as np
import imageio
import cv2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)#std

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return -(t * tf.log(o) + (1.- t)*tf.log(1. - o))

class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[64,64,3],
            dim_z=100,
            dim_W1=1024,
            dim_W2=512,
            dim_W3=256,
            dim_W4=128,
            dim_W5=3,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_W4 = dim_W4
        self.dim_W5 = dim_W5

        self.gen_W1 = tf.Variable(tf.random.normal([dim_z, dim_W1*4*4], stddev=0.02), name='gen_W1')
        self.gen_bn_g1 = tf.Variable( tf.random.normal([dim_W1*4*4], mean=1.0, stddev=0.02), name='gen_bn_g1')
        self.gen_bn_b1 = tf.Variable( tf.zeros([dim_W1*4*4]), name='gen_bn_b1')

        self.gen_W2 = tf.Variable(tf.random.normal([5,5,dim_W2, dim_W1], stddev=0.02), name='gen_W2')
        self.gen_bn_g2 = tf.Variable( tf.random.normal([dim_W2], mean=1.0, stddev=0.02), name='gen_bn_g2')
        self.gen_bn_b2 = tf.Variable( tf.zeros([dim_W2]), name='gen_bn_b2')

        self.gen_W3 = tf.Variable(tf.random.normal([5,5,dim_W3, dim_W2], stddev=0.02), name='gen_W3')
        self.gen_bn_g3 = tf.Variable( tf.random.normal([dim_W3], mean=1.0, stddev=0.02), name='gen_bn_g3')
        self.gen_bn_b3 = tf.Variable( tf.zeros([dim_W3]), name='gen_bn_b3')

        self.gen_W4 = tf.Variable(tf.random.normal([5,5,dim_W4, dim_W3], stddev=0.02), name='gen_W4')
        self.gen_bn_g4 = tf.Variable( tf.random.normal([dim_W4], mean=1.0, stddev=0.02), name='gen_bn_g4')
        self.gen_bn_b4 = tf.Variable( tf.zeros([dim_W4]), name='gen_bn_b4')

        self.gen_W5 = tf.Variable(tf.random.normal([5,5,dim_W5, dim_W4], stddev=0.02), name='gen_W5')

        self.discrim_W1 = tf.Variable(tf.random.normal([5,5,dim_W5,dim_W4], stddev=0.02), name='discrim_W1')

        self.discrim_W2 = tf.Variable(tf.random.normal([5,5,dim_W4,dim_W3], stddev=0.02), name='discrim_W2')
        self.discrim_bn_g2 = tf.Variable( tf.random.normal([dim_W3], mean=1.0, stddev=0.02), name='discrim_bn_g2')
        self.discrim_bn_b2 = tf.Variable( tf.zeros([dim_W3]), name='discrim_bn_b2')

        self.discrim_W3 = tf.Variable(tf.random.normal([5,5,dim_W3,dim_W2], stddev=0.02), name='discrim_W3')
        self.discrim_bn_g3 = tf.Variable( tf.random.normal([dim_W2], mean=1.0, stddev=0.02), name='discrim_bn_g3')
        self.discrim_bn_b3 = tf.Variable( tf.zeros([dim_W2]), name='discrim_bn_b3')

        self.discrim_W4 = tf.Variable(tf.random.normal([5,5,dim_W2,dim_W1], stddev=0.02), name='discrim_W4')
        self.discrim_bn_g4 = tf.Variable( tf.random.normal([dim_W1], mean=1.0, stddev=0.02), name='discrim_bn_g4')
        self.discrim_bn_b4 = tf.Variable( tf.zeros([dim_W1]), name='discrim_bn_b4')

        self.discrim_W5 = tf.Variable(tf.random.normal([4*4*dim_W1,1], stddev=0.02), name='discrim_W5')

        self.gen_params = [
                self.gen_W1, self.gen_bn_g1, self.gen_bn_b1,
                self.gen_W2, self.gen_bn_g2, self.gen_bn_b2,
                self.gen_W3, self.gen_bn_g3, self.gen_bn_b3,
                self.gen_W4, self.gen_bn_g4, self.gen_bn_b4,
                self.gen_W5
                ]

        self.discrim_params = [
                self.discrim_W1,
                self.discrim_W2, self.discrim_bn_g2, self.discrim_bn_b2,
                self.discrim_W3, self.discrim_bn_g3, self.discrim_bn_b3,
                self.discrim_W4, self.discrim_bn_g4, self.discrim_bn_b4,
                self.discrim_W5
                ]

    def build_model(self):

        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        image_gen = self.generate(Z)

        p_real, h_real = self.discriminate(image_real)
        p_gen, h_gen = self.discriminate(image_gen)

        discrim_cost_real = bce(p_real, tf.ones_like(p_real))
        discrim_cost_gen = bce(p_gen, tf.zeros_like(p_gen))
        discrim_cost = tf.reduce_mean(discrim_cost_real) + tf.reduce_mean(discrim_cost_gen)

        gen_cost = tf.reduce_mean(bce( p_gen, tf.ones_like(p_gen) ))

        return Z, image_real, discrim_cost, gen_cost, p_real, p_gen, h_real, h_gen

    def discriminate(self, image):
        h1 = lrelu( tf.nn.conv2d( image, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME'), g=self.discrim_bn_g2, b=self.discrim_bn_b2) )
        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.discrim_W3, strides=[1,2,2,1], padding='SAME'), g=self.discrim_bn_g3, b=self.discrim_bn_b3) )
        h4 = lrelu( batchnormalize( tf.nn.conv2d( h3, self.discrim_W4, strides=[1,2,2,1], padding='SAME'), g=self.discrim_bn_g4, b=self.discrim_bn_b4) )
        h4 = tf.reshape(h4, [self.batch_size, -1])
        h5 = tf.matmul( h4, self.discrim_W5 )
        y = tf.nn.sigmoid(h5)
        return y, h5

    def generate(self, Z):
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1), g=self.gen_bn_g1, b=self.gen_bn_b1))
        h1 = tf.reshape(h1, [self.batch_size,4,4,self.dim_W1])

        output_shape_l2 = [self.batch_size,8,8,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = tf.nn.relu( batchnormalize(h2, g=self.gen_bn_g2, b=self.gen_bn_b2) )

        output_shape_l3 = [self.batch_size,16,16,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3, g=self.gen_bn_g3, b=self.gen_bn_b3) )

        output_shape_l4 = [self.batch_size,32,32,self.dim_W4]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        h4 = tf.nn.relu( batchnormalize(h4, g=self.gen_bn_g4, b=self.gen_bn_b4) )

        output_shape_l5 = [self.batch_size,64,64,self.dim_W5]
        h5 = tf.nn.conv2d_transpose(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])

        x = tf.nn.tanh(h5)
        return x

    def samples_generator(self, batch_size):

        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.reshape(h1, [batch_size,4,4,self.dim_W1])

        output_shape_l2 = [batch_size,8,8,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = tf.nn.relu( batchnormalize(h2) )

        output_shape_l3 = [batch_size,16,16,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )

        output_shape_l4 = [batch_size,32,32,self.dim_W4]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        h4 = tf.nn.relu( batchnormalize(h4) )

        output_shape_l5 = [batch_size,64,64,self.dim_W5]
        h5 = tf.nn.conv2d_transpose(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])

        x = tf.nn.tanh(h5)
        return Z, x

def crop_resize(image_path, resize_shape=(64,64)):
    image = cv2.imread(image_path)
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

def save_visualization(X, nh, nw, save_path='./vis/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))
    for n,x in enumerate(X):
        j = np.floor(n / nw)
        i = n % nw
        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w), :] = x

    imageio.imwrite(save_path, img)

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [64,64,3]
dim_z = 100
dim_W1 = 1024
dim_W2 = 512
dim_W3 = 256
dim_W4 = 128
dim_W5 = 3

visualize_dim=196

face_image_path = "C:/celeba-dataset/img_align_celeba"
face_images = list(filter(lambda x: x.endswith('jpg'), os.listdir(face_image_path)))

dcgan_model = DCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        dim_W4=dim_W4,
        dim_W5=dim_W5
        )

Z_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, h_real, h_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=list(discrim_vars))
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=list(gen_vars))

Z_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.initialize_all_variables().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
iterations = 0
k = 2

for epoch in range(n_epochs):
    np.random.shuffle(face_images)

    for start, end in zip(
            range(0, len(face_images), batch_size),
            range(batch_size, len(face_images), batch_size)
            ):

        batch_image_files = face_images[start:end]
        batch_images = list(map(lambda x: crop_resize( os.path.join( face_image_path, x) ), batch_image_files))
        batch_images = np.array(batch_images).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

        if np.mod( iterations, k ) == 0:
            _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={
                        Z_tf:batch_z,
                        })
            discrim_loss_val, p_real_val, p_gen_val, h_real_val, h_gen_val = sess.run([d_cost_tf,p_real,p_gen, h_real, h_gen], feed_dict={Z_tf:batch_z, image_tf:batch_images})
            print("=========== updating G ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)

        else:
            _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        Z_tf:batch_z,
                        image_tf:batch_images
                        })
            gen_loss_val, p_real_val, p_gen_val, h_real_val, h_gen_val = sess.run([g_cost_tf, p_real, p_gen, h_real, h_gen], feed_dict={Z_tf:batch_z, image_tf:batch_images})
            print("=========== updating D ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)

        # ipdb.set_trace()

        if np.mod(iterations, 100) == 0:
            generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample:Z_np_sample
                        })
            generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, 14,14, save_path='C:/out/sample_'+str(iterations/100)+'.jpg')

        iterations += 1