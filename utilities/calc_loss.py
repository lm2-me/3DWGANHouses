"""
Functions to 

By: 
TensorFlow
https://github.com/tensorflow/tensorflow/blob/2007e1ba474030fcce840b0b8a599558e7d5998f/tensorflow/contrib/gan/python/losses/python/losses_impl.py


Classes:
    - None

Methods:
    >>> wasserstein_generator_loss: Wasserstein generator loss for GANs.
    >>> wasserstein_discriminator_loss: Wasserstein discriminator loss for GANs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary

# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
def wasserstein_generator_loss(
    discriminator_gen_outputs,
    num_vox,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein generator loss for GANs.
  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add detailed summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'generator_wasserstein_loss', (
      discriminator_gen_outputs, weights)) as scope:
    # discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)

    loss = - discriminator_gen_outputs
    loss = losses.compute_weighted_loss(
        loss, weights, scope, loss_collection, reduction)

    if add_summaries:
      summary.scalar('generator_wass_loss', loss)

    per_vox_loss = loss / num_vox
    
  return loss, per_vox_loss


def wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    num_vox,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein discriminator loss for GANs.
  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'discriminator_wasserstein_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights)) as scope:
    # discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
    # discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
    discriminator_real_outputs.shape.assert_is_compatible_with(
        discriminator_gen_outputs.shape)

    loss_on_generated = losses.compute_weighted_loss(
        discriminator_gen_outputs, generated_weights, scope,
        loss_collection=None, reduction=reduction)
    loss_on_real = losses.compute_weighted_loss(
        discriminator_real_outputs, real_weights, scope, loss_collection=None,
        reduction=reduction)
    loss = loss_on_generated - loss_on_real
    util.add_loss(loss, loss_collection)

    if add_summaries:
      summary.scalar('discriminator_gen_wass_loss', loss_on_generated)
      summary.scalar('discriminator_real_wass_loss', loss_on_real)
      summary.scalar('discriminator_wass_loss', loss)
    
    per_vox_loss = loss / num_vox
    per_vox_loss_real = loss_on_real / num_vox
    pre_vox_loss_fake = loss_on_generated / num_vox

  return loss, per_vox_loss, per_vox_loss_real, pre_vox_loss_fake
