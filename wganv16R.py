"""
Functions to generate and train GAN model

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Methods:
    >>> make_generator_model: Creates the generator model.
    >>> make_discriminator_model: Creates the discriminator (critic) model.
    >>> train: Training steps for the model.
    >>> train_gan: Creates required parameters for training and calls model training.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import initializers
import time
from tqdm import tqdm, trange

import utilities.ganutilities as util

np.set_printoptions(precision=4)

BUFFER_SIZE = 4 #how many models to load "on deck"
SAVE_NUM = 20
EPOCHS = 2000
INIT = initializers.HeNormal(seed=42)


#wgan v R is a gan using Wasserstein loss, LeakyReLU in the generator and discriminator, RMSProp with a set learning rate, 
# and gradient penalty implemented

#in wgans, the 'discriminator' is replaced by a critic but in the code it is still refered to as a discriminator to keep variable 
#names consistent across different architectures

#The Generator
def make_generator_model(num_class=2):
    '''
    Creates the generator model.

            Parameters:
                    num_class: the number of classes used when encoding the data set into numpy arrays (also the dimension of axis 3 of the array) 

            Returns:
                    model: the generator model
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(10 * 10 * 5 * 96, use_bias=False, input_shape=(200,), kernel_initializer=INIT))

    model.add(layers.Reshape((10, 10, 5, 96)))
    assert model.output_shape == (None, 10, 10, 5, 96)

    model.add(layers.Conv3DTranspose(96, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 10, 10, 5, 96)

    model.add(layers.Conv3DTranspose(48, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 20, 20, 10, 48)

    model.add(layers.Conv3DTranspose(48, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 20, 20, 10, 48)
    
    model.add(layers.Conv3DTranspose(24, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 40, 40, 20, 24)

    model.add(layers.Conv3DTranspose(24, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 40, 40, 20, 24)

    model.add(layers.Conv3DTranspose(12, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 80, 80, 40, 12)

    model.add(layers.Conv3DTranspose(12, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 80, 80, 40, 12)

    model.add(layers.Conv3DTranspose(num_class, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 160, 160, 80, num_class)

    model.add(layers.Conv3DTranspose(num_class, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 160, 160, 80, num_class)

    return model

#The Discriminator
def make_discriminator_model(num_class=2):
    '''
    Creates the discriminator (critic) model.

            Parameters:
                    num_class: the number of classes used when encoding the data set into numpy arrays (also the dimension of axis 3 of the array) 

            Returns:
                    model: the discriminator model
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(12, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT,
                                     input_shape=[160, 160, 80, num_class]))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 80, 80, 40, 12)

    model.add(layers.Conv3D(12, (4, 4, 4), strides=(1, 1, 1), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 80, 80, 40, 12)

    model.add(layers.Conv3D(24, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 40, 40, 20, 24)

    model.add(layers.Conv3D(24, (4, 4, 4), strides=(1, 1, 1), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 40, 40, 20, 24)

    model.add(layers.Conv3D(48, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 20, 20, 10, 48)

    model.add(layers.Conv3D(48, (4, 4, 4), strides=(1, 1, 1), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 20, 20, 10, 48)

    model.add(layers.Conv3D(1, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 10, 10, 5, 1)

    model.add(layers.Conv3D(1, (4, 4, 4), strides=(1, 1, 1), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 10, 10, 5, 1)

    model.add(layers.Flatten())
    model.add(layers.Dense(1, use_bias=False, input_shape=(200,), kernel_initializer=INIT))
    
    return model
    
def train(run_name, dataset, epochs, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, seed, acc_real, acc_fake, num_vox):
  '''
  Training steps for the model.

          Parameters:
                  run_name: name of current training run
                  dataset: loaded data set
                  epochs: number of epochs to train
                  noise_dim: dimension of the seed
                  generator: the generator model
                  discriminator: the discriminator (critic) model
                  generator_optimizer: tensorflow optimizer for the generator
                  discriminator_optimizer: tensorflow optimizer for the discriminator (critic)
                  seed: tensorflow normalized random tensor
                  acc_real: tensorflow binary accuracy metric to track accuracy of real geometry
                  acc_fake: tensorflow binary accuracy metric to track accuracy of fake geometry
                  num_vox: total number of voxels in the geometry space
          Returns:
                  none
  '''
   
  for epoch in trange(epochs, desc='Epoch'):
    start = time.time()

    for image_batch in tqdm(dataset, desc='Iteration'):

      gen_per_vox_loss = util.train_generator(noise_dim, generator, discriminator, generator_optimizer, num_vox)
      real_loss, disc_per_vox_loss, fake_loss, real_acc, fake_acc, gradients_of_discriminator, gradient_penalty = util.train_discriminator(image_batch, noise_dim, generator, discriminator, acc_real, acc_fake, num_vox, use_grad_pen=True)

      util.update_discriminator(discriminator_optimizer, gradients_of_discriminator, discriminator)
   
      util.print_loss_info('generated', run_name, epoch, gen_per_vox_loss, real_loss, fake_loss, disc_per_vox_loss, real_acc, fake_acc)

      acc_real.reset_states()
      acc_fake.reset_states()

    # Save generated matrices 
    if epoch % (EPOCHS//SAVE_NUM) == 0:
      util.save_generated_matrix(run_name, generator, epoch, seed, testID = run_name)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Save after the final epoch
  util.save_generated_matrix(run_name, generator, epoch, seed, testID = run_name)

def train_gan(path, run_name, num_vox):
    '''
    Creates required parameters for training and calls model training.

            Parameters:
                    path: path of the training data
                    run_name: name of current training run
                    num_vox: total number of voxels in the geometry space
            Returns:
                    none
    '''
    
    train_dataset = util.load_dataset(path)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    acc_real = tf.keras.metrics.BinaryAccuracy()
    acc_fake = tf.keras.metrics.BinaryAccuracy()

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    
    #define training loop
    noise_dim = 200
    num_examples_to_generate = 1

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    train(run_name, train_dataset, EPOCHS, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, seed, acc_real, acc_fake, num_vox)



