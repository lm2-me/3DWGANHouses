"""
functions supporting GAN traning and visualizations

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - ReflectionPadding3D: generates padding that reflects the matrix values instead of using 0 values

Methods:
	  >>> load_dataset: loads the data set based on the buffer 
    >>> save_generated_matrix: saves the output of the generator to a numpy array
    >>> gen_loss_steps: calculates the generator's loss
    >>> train_generator: the training steps for the generator
    >>> disc_loss_steps: calculates the discriminator's (critic's) loss
    >>> train_discriminator: the training steps for the discriminator (critic)
    >>> update_discriminator: updates the discirminator (critic) model
	  >>> discriminator_accuracy: calculates the accuracy of the discriminator (critic)
    >>> print_loss_info: saves the loss information to a csv file
    >>> learning_rate_decay: calculates the updated learning rate based on the decay factor
    >>> set_learning_rate: sets the new learning rate based on the decay factor
    >>> visualize_files_from_folder: generates images of matrices from a directory
    >>> list_of_files: returns all files in a directory
    >>> _load_encoded_array: loads the numpy array
    >>> render_gif: generates a voxel image of a numpy array
    >>> get_color_norm: returns the color to use for visualizations
    
"""

import tensorflow as tf
import tensorflow.keras.backend as bk
from tensorflow.keras import layers
import math

from tensorflow.python.ops.losses import losses
import os, sys
import numpy as np
from datetime import datetime

from simple_3dviz import Mesh
from simple_3dviz.window import show
from simple_3dviz.utils import render
from simple_3dviz.scenes import Scene
from PIL import Image


import utilities.calc_loss as loss
import utilities.ganloadsave as ldsv

np.set_printoptions(precision=4)

BUFFER_SIZE = 4 #how many models to load "on deck"
BATCH_SIZE = 4 #1-4 batch size (this should never be greater than the training dataset)

MIN_LR = 0.000001 # Minimum value of learning rate
DECAY_FACTOR=1.00004 # learning rate decay factor

STATIC_RESOLUTION = 4096

#gradient penalty 
LAMBDA = 10

### Padding
class ReflectionPadding3D(layers.Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [layers.InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        #for "channels_last" configuration which is the default
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x, mask=None):
        w_pad,h_pad,d_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0,0] ], 'REFLECT')

### Dataset and Matrices

def load_dataset(path):
    train_data = ldsv.load_data_for_gan_individually(path)

    # Batch and shuffle the data
    train_dataset = train_data.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_dataset

def save_generated_matrix(run_name, model, epoch, test_input, testID = None):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  
  predictions = model(test_input, training=False)

  generatored_geometry = predictions.numpy()

  testID = run_name
  
  matrix_path = os.path.join('generated/generated_matrices', testID)
  os.makedirs(matrix_path, exist_ok=True)

  for i in range(test_input.shape[0]):
    name = str(epoch) + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(i) + 'generated.npy'
    save_location = os.path.join(matrix_path, name)
    ldsv.save_encode_array(save_location, generatored_geometry[i])

### Training

@tf.function
def gen_loss_steps(noise, generator, discriminator, num_vox):
  generated_models = generator(noise, training=True)
  fake_output = discriminator(generated_models, training=True)

  #! also try other reduction
  # gen_loss, gen_loss_per_vox = loss.wasserstein_generator_loss(fake_output, num_vox, reduction=losses.Reduction.MEAN) 
  gen_loss, gen_loss_per_vox = loss.wasserstein_generator_loss(fake_output, num_vox)

  return gen_loss, gen_loss_per_vox

def train_generator(noise_dim, generator, discriminator, generator_optimizer, num_vox):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape:
      gen_loss, gen_loss_per_vox = gen_loss_steps(noise, generator, discriminator, num_vox)
      
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss_per_vox

@tf.function
def disc_loss_steps(models, noise, epsilon, generator, discriminator, num_vox, acc_real, acc_fake, grad_pen):

    generated_models = generator(noise, training=True)

    if grad_pen:
      fake_models_mixed = epsilon * tf.dtypes.cast(models, tf.float32) + ((1 - epsilon) * generated_models)

      with tf.GradientTape() as gp_tape:
        gp_tape.watch(fake_models_mixed)
        fake_model_pred = discriminator(fake_models_mixed, training=True)

      #gradient penalty
      grads = gp_tape.gradient(fake_model_pred, fake_models_mixed)
      grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
      gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
    
    else: 
      gradient_penalty = 0

    real_output = discriminator(models, training=True)
    fake_output = discriminator(generated_models, training=True)

    #! try with other reduction
    
    #total_loss, per_vox_loss, real_loss, fake_loss = loss.wasserstein_discriminator_loss(real_output, fake_output, num_vox, reduction=losses.Reduction.MEAN)
    total_loss, per_vox_loss, real_loss, fake_loss = loss.wasserstein_discriminator_loss(real_output, fake_output, num_vox)
    real_acc, fake_acc = discriminator_accuracy(real_output, fake_output, acc_real, acc_fake)

    total_loss = total_loss + LAMBDA * gradient_penalty
    per_vox_loss = total_loss / num_vox

    return total_loss, per_vox_loss, real_loss, fake_loss, real_acc, fake_acc, gradient_penalty 

def train_discriminator(models, noise_dim, generator, discriminator, acc_real, acc_fake, num_vox, use_grad_pen=False):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    epsilon = tf.random.uniform(shape = [BATCH_SIZE, 1, 1, 1, 1], minval=0, maxval=1)

    with tf.GradientTape(persistent=True) as disc_tape:
      total_loss, per_vox_loss, real_loss, fake_loss, real_acc, fake_acc, gradient_penalty = disc_loss_steps(models, noise, epsilon, generator, discriminator, num_vox, acc_real, acc_fake, use_grad_pen)
      
    gradients_of_discriminator = disc_tape.gradient(total_loss, discriminator.trainable_variables)

    return real_loss, per_vox_loss, fake_loss, real_acc, fake_acc, gradients_of_discriminator, gradient_penalty

@tf.function
def update_discriminator(discriminator_optimizer, gradients_of_discriminator, discriminator):
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

### Accuracy

def discriminator_accuracy(real_output, fake_output, acc_real, acc_fake):
    acc_real.update_state(tf.ones_like(real_output), real_output)
    acc_fake.update_state(tf.zeros_like(fake_output), fake_output)
    
    real_acc_result = acc_real.result()
    fake_acc_result = acc_fake.result()

    return real_acc_result, fake_acc_result

### Loss

def print_loss_info(path, run_name, epoch, gen_loss, real_loss, fake_loss, total_loss, real_acc, fake_acc):
    gen_loss_float = bk.eval(gen_loss)
    real_loss_float = bk.eval(real_loss)
    fake_loss_float = bk.eval(fake_loss)
    total_loss_float = bk.eval(total_loss)

    epoch_acc_real_float = bk.eval(real_acc)
    epoch_acc_fake_float = bk.eval(fake_acc)

    #tf.print("Generator loss: {}. Discriminator real loss: {}. Discriminator fake loss: {}.".format(gen_loss_float, real_loss_float, fake_loss_float), output_stream=sys.stdout)
    #tf.print("Discriminator real accuracy: {} Discriminator fake accuracy {}".format(epoch_acc_real_float, epoch_acc_fake_float))

    if not os.path.isdir(path + '/csv'):
      os.makedirs(path + '/csv')
      print('created folder at {}'.format(str(path + '/csv')))

    row = [epoch, gen_loss_float, real_loss_float, fake_loss_float, total_loss_float, epoch_acc_real_float, epoch_acc_fake_float]
    
    ldsv._save_loss_to_csv(path + '/csv/' + run_name + '.csv', row)

### Learning Rate

def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, generator_optimizer, discriminator_optimizer):
    bk.set_value(discriminator_optimizer.lr, new_lr)
    bk.set_value(generator_optimizer.lr, new_lr)


# Visualizing Results

def visualize_files_from_folder(load_path, save_folder, run_name):

  all_files = list_of_files(load_path)
  print('located {} files in {}'.format(len(all_files), load_path))

  save_path = os.path.join(save_folder, run_name)
  generated = 0

  if not os.path.isdir(save_path):
      os.makedirs(save_path)
      print('created folder at {}'.format(str(save_path)))

  existing_files = list_of_files(save_path)
  print('{} images were already visualized. Generating {} images, please wait'.format(len(existing_files), len(all_files)-len(existing_files)))


  for file in all_files:
    if file.endswith('.npy') and ('encode' in file or 'generated' in file):
      name = file.replace('.npy','')

      if not os.path.isfile(os.path.join(save_path, name + '.png')):
        generated += 1
        encoded_array = _load_encoded_array(load_path + '/' + file)

        save_file = os.path.join(save_path, name + '.png') 
        render_gif(encoded_array, output_path=save_file)
  
  print('generated {} files in {}'.format(generated, load_path))

def list_of_files(path_loc):
  all_files = [f for f in os.listdir(path_loc) if os.path.isfile(os.path.join(path_loc, f))]
  return all_files

def _load_encoded_array(pathname):
  with open(pathname, 'rb') as f:
      data = np.load(f)
  if len(data.shape) == 5:
      return data[0]
  else:
      return data 
    
def render_gif(voxel_space, output_path=None):
  #print('space shape', voxel_space.shape)
  
  classes = voxel_space.shape[3]
  arr_x = voxel_space.shape[0]
  arr_y = voxel_space.shape[1]
  arr_z = voxel_space.shape[2]
  max_dim = max(arr_x, arr_y, arr_z)
  voxel_space_class = np.zeros((max_dim, max_dim, max_dim))
  center_point = max_dim // 2 

  voxel_space_class[
    (center_point - math.floor(arr_x / 2.)):(center_point + math.ceil(arr_x / 2.)),
    (center_point - math.floor(arr_y / 2.)):(center_point + math.ceil(arr_y / 2.)),
    (center_point - math.floor(arr_z / 2.)):(center_point + math.ceil(arr_z / 2.))
  ] = np.argmax(voxel_space, axis=3)
  
  #print('Computed voxel space')
  
  if np.count_nonzero(voxel_space_class > 0) == 0:
    print('Empty voxel space')
    return
  
  meshes = [Mesh.from_voxel_grid(voxel_space_class == i, colors=get_color_norm(i-1)) for i in range(1, classes)]
  
  #print('Computed meshes')
  
  #!usual settings
  camera_distance = 2.
  camera_height = 1

  #!changed settings
  # camera_distance = 2.
  # camera_height = .5


  #! usual settings
  th = 3 * math.pi / 4. # view from 45 degrees

  #!changed settings
  #th = 7 * math.pi / 4. # view from 45 degrees other size
  #th = -1 * math.pi / 8
  
  x = camera_distance * math.sin(th)
  y = camera_distance * math.cos(th)

  if output_path is None:
    show(meshes,
        background=(1, 1, 1, 1),
        camera_position=(x, y, camera_height),
        light = (0.5, 0.8, 2))

  else:
    scene = Scene(size=(STATIC_RESOLUTION, STATIC_RESOLUTION),
                  background=(1, 1, 1, 1))
    for m in meshes:
      scene.add(m)
    
    scene.camera_position = (x, y, camera_height)
    scene.light = (0.5, 0.8, 2)
    scene.render()

    im = Image.fromarray(scene.frame)
    im.save(output_path)

def get_color_norm(index):
    colors_1 = [(0.9019607843137255, 0.09803921568627451, 0.29411764705882354), (0.23529411764705882, 0.7058823529411765, 0.29411764705882354), (1.0, 0.8823529411764706, 0.09803921568627451), (0.2627450980392157, 0.38823529411764707, 0.8470588235294118), (0.9607843137254902, 0.5098039215686274, 0.19215686274509805), (0.5686274509803921, 0.11764705882352941, 0.7058823529411765), (0.25882352941176473, 0.8313725490196079, 0.9568627450980393), (0.9411764705882353, 0.19607843137254902, 0.9019607843137255), (0.7490196078431373, 0.9372549019607843, 0.27058823529411763), (0.9803921568627451, 0.7450980392156863, 0.8313725490196079), (0.27450980392156865, 0.6, 0.5647058823529412), (0.8627450980392157, 0.7450980392156863, 1.0), (0.6039215686274509, 0.38823529411764707, 0.1411764705882353), (1.0, 0.9803921568627451, 0.7843137254901961), (0.5019607843137255, 0.0, 0.0), (0.6666666666666666, 1.0, 0.7647058823529411), (0.5019607843137255, 0.5019607843137255, 0.0), (1.0, 0.8470588235294118, 0.6941176470588235), (0.0, 0.0, 0.4588235294117647), (0.6627450980392157, 0.6627450980392157, 0.6627450980392157), (0.6, 0.6, 0.6), (0.0, 0.0, 0.0)]
    colors_2 = [(0.6980392156862745, 0.0, 0.0), (0.6666666666666666, 1.0, 0.0), (0.0, 0.0, 1.0), (0.4, 0.0, 0.26666666666666666), (0.21176470588235294, 0.7019607843137254, 0.5372549019607843), (0.12156862745098039, 0.12156862745098039, 0.4), (0.0, 0.30196078431372547, 0.2), (0.8, 0.23921568627450981, 0.611764705882353), (0.0, 0.0, 0.2), (0.23921568627450981, 0.23921568627450981, 0.8)]

    colors = colors_1 + colors_2
    return colors[index]
