import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras import initializers
import time
from tqdm import tqdm, trange

import gan_utils.ganutilities_cubes as util

np.set_printoptions(precision=4)

BUFFER_SIZE = 4 #how many models to load "on deck"
SAVE_NUM = 40

EPOCHS = 8000
INIT = initializers.HeNormal(seed=42)


#wgan v R is a gan using Wasserstein loss, LeakyReLU, RMSProp, and with gradient penalty (no normalization and no learning rate decay)

#The Generator
def make_generator_model(num_class=2):
    model = tf.keras.Sequential()
    #model.add(layers.BatchNormalization())
    model.add(layers.Input((10, 10, 5, 192)))
    assert model.output_shape == (None, 10, 10, 5, 192)

    model.add(layers.Conv3DTranspose(96, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 20, 20, 10, 96)
    
    model.add(layers.Conv3DTranspose(48, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 40, 40, 20, 48)

    model.add(layers.Conv3DTranspose(24, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=INIT))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 80, 80, 40, 24)

    model.add(layers.Conv3DTranspose(num_class, (4, 4, 4), strides=(2, 2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 160, 160, 80, num_class)

    return model

#The Discriminator
def make_discriminator_model(num_class=2):
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(24, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT,
                                     input_shape=[160, 160, 80, num_class]))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 80, 80, 40, 24)

    model.add(layers.Conv3D(48, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 40, 40, 20, 48)

    model.add(layers.Conv3D(96, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 20, 20, 10, 96)

    model.add(layers.Conv3D(1, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=INIT))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 10, 10, 5, 1)

    model.add(layers.Flatten())
    model.add(layers.Dense(1, use_bias=False, input_shape=(200,), kernel_initializer=INIT))
    
    return model
    
def train(path, run_name, dataset, epochs, inputsize, generator, discriminator, generator_optimizer, discriminator_optimizer, seed, checkpoint_manager, acc_real, acc_fake, num_vox):
  
  for epoch in trange(epochs, desc='Epoch'):
    start = time.time()

    for image_batch in tqdm(dataset, desc='Iteration'):

      gen_per_vox_loss = util.train_generator(inputsize, generator, discriminator, generator_optimizer, num_vox)

      real_loss, disc_per_vox_loss, fake_loss, real_acc, fake_acc, gradients_of_discriminator, gradient_penalty = util.train_discriminator(image_batch, inputsize, generator, discriminator, acc_real, acc_fake, num_vox, use_grad_pen=True)

      av_accuracy = (real_acc + fake_acc) / 2
      #update discriminator only if the accuracy is < 0.8
      check_result = util.check_discriminator_accuracy(av_accuracy)
      if check_result: 
        util.update_discriminator(discriminator_optimizer, gradients_of_discriminator, discriminator)
   
      util.print_loss_info(path, run_name, epoch, gen_per_vox_loss, real_loss, fake_loss, disc_per_vox_loss, real_acc, fake_acc, check_result)

      acc_real.reset_states()
      acc_fake.reset_states()

    # Save matrices 
    if epoch % (EPOCHS//SAVE_NUM) == 0:

      util.save_generated_matrix(path, run_name, generator, epoch, seed, testID = run_name)

    # Save the model every 15 epochs
    if (epoch + 1) % EPOCHS//SAVE_NUM == 0:
      checkpoint_manager.save()

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Save after the final epoch
  util.save_generated_matrix(path, run_name, generator, epoch, seed, testID = run_name)

def train_gan(path, run_name, num_vox, max_load, full, twohundred):
    
    train_dataset = util.load_dataset(path, max_load, full, twohundred)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    acc_real = tf.keras.metrics.BinaryAccuracy()
    acc_fake = tf.keras.metrics.BinaryAccuracy()

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    #Save checkpoints in case long running training tasks are interrupted
    checkpoint_dir = os.path.join(path, 'training_checkpoints', run_name)

    if not os.path.isdir(checkpoint_dir):
      os.makedirs(checkpoint_dir)
      load_from_checkpoint = False
      print('created folder at {}'.format(str(checkpoint_dir)))
    
    else:
      load_from_checkpoint = True

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep = 3)

    if load_from_checkpoint:
      checkpoint.restore(manager.latest_checkpoint)
      print("Checkpoint restored")
    
    #define training loop
    num_examples_to_generate = 1
    inputsize = [10, 10, 5, 192]
    #! update 10, 10 , 5 to be same size as first generator input

    cube_input = util.generate_random_cube(num_examples_to_generate, inputsize)

    train(path, run_name, train_dataset, EPOCHS, inputsize, generator, discriminator, generator_optimizer, discriminator_optimizer, cube_input, manager, acc_real, acc_fake, num_vox)



