import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import PIL
import keras.api._v2.keras as keras
from keras import layers
from keras import initializers
import time
from datetime import datetime
from tqdm import tqdm, trange
import pathlib
import pandas as pd
import keras.backend as bk

from IPython import display

import visualization.visualize as vis
import preprocessing.loadsave as ldsv
import preprocessing.encode as encode

np.set_printoptions(precision=4)

BUFFER_SIZE = 10
BATCH_SIZE = 1 #1-4 batch size (this should never be greater than the training dataset)
EPOCHS = 8000 
SAVE_NUM = 40
INIT = initializers.HeNormal() 

#The base architecture for DCGAN is described as follows:
'''This architecture is based on the precedent of 3D GAN architecture [Wu et al., 2016].
LOSS FUNCTION: Min-Max GAN loss function
GENERATOR: 4 fully convolution layers with a number of channels of {48, 24, 12, 2}, kernel sizes {4, 4, 4, 4}, and strides {2, 2, 2, 2};
activation function ReLU, and Softmax at the end; input 200-dimensional vector, output is a [160, 160, 80] matrix with values in [0, 1];
ADAM optimizer with Learning Rate=0.0025 and Beta=0.5

DISCRIMINATOR: mirrored version of the generator
input of a [160, 160, 80] matrix, outputs a real number in [0, 1]; 4 volumetric convolution layers, with a number of channels 
of {12, 24, 48, 2, kernel sizes {4, 4, 4, 4}, and strides {2, 2, 2, 2}; activation function Leaky ReLU alpha = 0.2, and a
Sigmoid layer at the end; ADAM optimizer with Learning Rate=0.00001 and Beta=0.5'''

#The Generator
def make_generator_model(num_class=2):
    model = tf.keras.Sequential()
    model.add(layers.Dense(20 * 20 * 10 * 48, use_bias=False, input_shape=(200,), kernel_initializer=INIT))
    #model.add(layers.BatchNormalization())

    model.add(layers.Reshape((20, 20, 10, 48)))
    assert model.output_shape == (None, 20, 20, 10, 48)

    model.add(layers.Conv3DTranspose(24, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 40, 40, 20, 24)
    
    model.add(layers.Conv3DTranspose(12, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 80, 80, 40, 12)

    model.add(layers.Conv3DTranspose(num_class, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, activation='softmax'))
    assert model.output_shape == (None, 160, 160, 80, num_class)

    return model

def make_discriminator_model(num_class=2):
    model = tf.keras.Sequential()

    model.add(layers.Conv3D(12, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT,
                                     input_shape=[160, 160, 80, num_class]))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 80, 80, 40, 12)

    model.add(layers.Conv3D(24, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 40, 40, 20, 24)

    model.add(layers.Conv3D(1, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='sigmoid'))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 20, 20, 10, 1)

    return model

def discriminator_loss(real_output, fake_output, cross_entropy):
    ones = tf.random.uniform(real_output.shape, minval=0.7, maxval=1.2)
    zeros = tf.random.uniform(fake_output.shape, minval=0.0, maxval=0.3)
    real_loss = cross_entropy(ones, real_output)
    fake_loss = cross_entropy(zeros, fake_output)
    #total loss is average
    total_loss = (real_loss + fake_loss) / 2
    return total_loss, real_loss, fake_loss

def discriminator_accuracy(real_output, fake_output, acc_real, acc_fake):
    # tf.print('-------------------------------------')
    # tf.print('real output')
    # tf.print(real_output)
    # tf.print('fake output')
    # tf.print(fake_output)

    acc_real.update_state(tf.ones_like(real_output), real_output)
    acc_fake.update_state(tf.zeros_like(fake_output), fake_output)
    
    real_acc_result = acc_real.result()
    fake_acc_result = acc_fake.result()

    return real_acc_result, fake_acc_result

def generator_loss(fake_output, cross_entropy):
    ones = tf.random.uniform(fake_output.shape, minval=0.7, maxval=1.2)
    return cross_entropy(ones, fake_output)

@tf.function
def train_generator(noise_dim, generator, discriminator, generator_optimizer, cross_entropy):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape:
      generated_models = generator(noise, training=True)
      fake_output = discriminator(generated_models, training=True)
      gen_loss = generator_loss(fake_output, cross_entropy)
      
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss

@tf.function
def train_discriminator(models, noise_dim, generator, discriminator, cross_entropy, acc_real, acc_fake):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as disc_tape:
      generated_models = generator(noise, training=True)

      real_output = discriminator(models, training=True)
      fake_output = discriminator(generated_models, training=True)

      total_loss, real_loss, fake_loss = discriminator_loss(real_output, fake_output, cross_entropy)
      real_acc, fake_acc = discriminator_accuracy(real_output, fake_output, acc_real, acc_fake)
      
    gradients_of_discriminator = disc_tape.gradient(total_loss, discriminator.trainable_variables)

    return real_loss, fake_loss, total_loss, real_acc, fake_acc, gradients_of_discriminator

@tf.function
def update_discriminator(discriminator_optimizer, gradients_of_discriminator, discriminator):
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def check_discriminator_accuracy(train_acc):
  if train_acc < 0.9:
    return True
  else:
    return True

def print_loss_info(run_name, epoch, gen_loss, real_loss, fake_loss, total_loss, real_acc, fake_acc, check_result):
    gen_loss_float = bk.eval(gen_loss)
    real_loss_float = bk.eval(real_loss)
    fake_loss_float = bk.eval(fake_loss)
    total_loss_float = bk.eval(total_loss)

    
    epoch_acc_real_float = bk.eval(real_acc)
    epoch_acc_fake_float = bk.eval(fake_acc)

    tf.print("Generator loss: {}. Discriminator real loss: {}. Discriminator fake loss: {}.".format(gen_loss_float, real_loss_float, fake_loss_float), output_stream=sys.stdout)
    tf.print("Discriminator real accuracy: {} Discriminator fake accuracy {}".format(epoch_acc_real_float, epoch_acc_fake_float))

    path = 'dataset/test_set_onestory'
    if not os.path.isdir(path + '/csv'):
      os.makedirs(path + '/csv')
      print('created folder at {}'.format(str(path + '/csv')))

    if check_result:
      row = [epoch, gen_loss_float, real_loss_float, fake_loss_float, total_loss_float, epoch_acc_real_float, epoch_acc_fake_float, 'x']
    
    else: 
      row = [epoch, gen_loss_float, real_loss_float, fake_loss_float, total_loss_float, epoch_acc_real_float, epoch_acc_fake_float, 'o']
    
    ldsv._save_loss_to_csv(path + '/csv/' + run_name + '.csv', row)
    
def train(run_name, dataset, epochs, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, seed, checkpoint, checkpoint_prefix, cross_entropy, acc_real, acc_fake):
  for epoch in trange(epochs, desc='Epoch'):
    start = time.time()

    for image_batch in tqdm(dataset, desc='Iteration'):

      gen_loss = train_generator(noise_dim, generator, discriminator, generator_optimizer, cross_entropy)
      real_loss, fake_loss, total_loss, real_acc, fake_acc, gradients_of_discriminator = train_discriminator(image_batch, noise_dim, generator, discriminator, cross_entropy, acc_real, acc_fake)

      av_accuracy = (real_acc + fake_acc) / 2
      #update discriminator only if the accuracy is < 0.8
      check_result = check_discriminator_accuracy(av_accuracy)
      if check_result: 
        update_discriminator(discriminator_optimizer, gradients_of_discriminator, discriminator)
         
      print_loss_info(run_name, epoch, gen_loss, real_loss, fake_loss, total_loss, real_acc, fake_acc, check_result)

      acc_real.reset_states()
      acc_fake.reset_states()

    # Save matrices
    if epoch % (EPOCHS//SAVE_NUM) == 0:

      save_generated_matrix(run_name, generator, epoch, seed, testID = run_name)

    if (epoch + 1) % EPOCHS//SAVE_NUM == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Save matrix after the final epoch
  save_generated_matrix(run_name, generator, epoch, seed, testID = run_name)

def save_generated_matrix(run_name, model, epoch, test_input, testID = None):
  predictions = model(test_input, training=False)

  generatored_geometry = predictions.numpy()

  path = 'dataset/test_set_onestory'
  testID = run_name
  
  if not os.path.isdir(path +'/matrices'):
      os.makedirs(path +'/matrices')
      print('created folder at {}'.format(str(path +'/matrices')))
  
  if not os.path.isdir(path +'/matrices'+'/'+testID):
      os.makedirs(path +'/matrices'+'/'+testID)
      print('created folder at {}'.format(str(path +'/matrices'+'/'+testID)))

  for i in range(test_input.shape[0]):
    name = str(epoch) + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(i)
    save_location = path +'/matrices'+'/'+testID
    ldsv.save_encode_array(save_location + '/' + name + 'generated.npy', generatored_geometry[i])

def train_gan(run_name, num_vox):

    path = 'dataset/test_set_onestory'
    
    train_data = ldsv.load_data_for_gan(path, max_files_load=1, smooth=False)

    #Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    acc_real = tf.keras.metrics.BinaryAccuracy()
    acc_fake = tf.keras.metrics.BinaryAccuracy()

    #Learning rates based on Wu, et al.
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0025, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5)

    #Save checkpoints in case long running training tasks are interrupted
    checkpoint_dir = 'dataset/test_set_onestory/training_checkpoints' + '/' + run_name

    if not os.path.isdir(checkpoint_dir):
      os.makedirs(checkpoint_dir)
      print('created folder at {}'.format(str(checkpoint_dir)))

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    noise_dim = 200
    num_examples_to_generate = 1

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train(run_name, train_dataset, EPOCHS, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, seed, checkpoint, checkpoint_prefix, cross_entropy, acc_real, acc_fake)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

