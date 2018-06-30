
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import math

from six.moves import urllib
import tensorflow as tf

import cifar_input

FLAGS = tf.app.flags.FLAGS
    
# Basic model parameters.    
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'c:\\temp\\cifar100_data',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-100 data set.
IMAGE_SIZE = 32
NUM_CLASSES = cifar_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

NUM_EXMPLES_PER_FOR_TRAIN = 1000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.\

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

GROWTH_RATE = 12
DEPTH_DENESE = 8
REDUCION = 0.5


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
  Returns:
        Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def add_dense_layer(input_layer ,name):
    with tf.variable_scope(name) as scope:
        for i in range(DEPTH_DENESE):
            # conv 1X1
            in_channel = input_layer.get_shape().as_list()[3] 
            batch1 = tf.layers.batch_normalization(input_layer)
            relu1 = tf.nn.relu(batch1)
            kernel1 = _variable_with_weight_decay('weights%s_1' % i, [1, 1, in_channel, 4 * GROWTH_RATE], 5e-2, None)
            conv1 = tf.nn.conv2d(relu1, kernel1, [1, 1, 1, 1], padding='SAME')
    
            # conv 3X3
            batch2= tf.layers.batch_normalization(conv1)
            relu2 = tf.nn.relu(batch2)
            kernel2 = _variable_with_weight_decay('weights%s_2' % i, [3, 3, 4 * GROWTH_RATE,  GROWTH_RATE], 5e-2, None)
            conv2 = tf.nn.conv2d(relu2, kernel2, [1, 1, 1, 1], padding='SAME')
    
            # concat output layer to input layer
            input_layer = tf.concat([conv2, input_layer], axis=3)

    return input_layer


def add_transition_layer(input_layer, name, last=False, pool_size=2):
    with tf.variable_scope(name) as scope:
          # BN and Relu
          in_channel = input_layer.get_shape().as_list()[3] 
          output_size = math.floor(in_channel * REDUCION)
          batch = tf.layers.batch_normalization(input_layer)
          relu = tf.nn.relu(batch)
          if last:
                # average pooling
                pooling = tf.nn.avg_pool(relu, [1, pool_size, pool_size, 1], [1, 1, 1, 1], padding='VALID')
          else:
                # Conv 1X1
                kernel = _variable_with_weight_decay('weights', [1, 1, in_channel, output_size], 5e-2, None)
                conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
        
                # max pooling
                pooling = tf.nn.max_pool(conv, [1, pool_size, pool_size, 1], [1, 2, 2, 1], padding='SAME')

    return pooling

def inputs():
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-100-binary')
    images, labels = cifar_input.inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
    return images, labels

def inference(images):
    """Build the CIFAR-10 model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """

    # conv
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', [3, 3, 3, GROWTH_RATE], 5e-2, None)
        conv1 = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

    # Dense Block 1
    dense1_layer = add_dense_layer(conv1, 'dense1')
    trans1_layer = add_transition_layer(dense1_layer, 'trans1')

    # Dense Block 2
    dense2_layer = add_dense_layer(trans1_layer, 'dense2')
    trans2_layer = add_transition_layer(dense2_layer, 'trans2')

    # Dense Block 3
    dense3_layer = add_dense_layer(trans2_layer, 'dense3')
    trans3_layer = add_transition_layer(dense3_layer, 'trans3', True, 8)

    # Fully connected
    with tf.variable_scope('fullyConnected') as scope:
        old_shape = [int(s) for s in trans3_layer.shape]
        trans3_layer_shapes = tf.reshape(trans3_layer, [old_shape[0], old_shape[1] * old_shape[2] * old_shape [3]])
        fully_connected = tf.contrib.layers.fully_connected(trans3_layer_shapes, 100, scope='fullyConnected')

    return fully_connected


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean


def train(total_loss, global_step):
    """Train CIFAR-100 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
          total_loss: Total loss from loss().
          global_step: Integer Variable counting the number of training steps
         processed.
    Returns:
       train_op: op for training.
    """
    # Determine learning rate
    '''
    lr = tf.Variable(0, trainable=False)
    boundaries = [int(NUM_EXMPLES_PER_FOR_TRAIN * 0.5), int(NUM_EXMPLES_PER_FOR_TRAIN * 0.75)]
    values = [INITIAL_LEARNING_RATE, NUM_EXMPLES_PER_FOR_TRAIN / 10 , NUM_EXMPLES_PER_FOR_TRAIN / 100]
    tf.train.piecewise_constant(lr, boundaries, values)
    '''
    # Compute gradients.
    opt = tf.train.GradientDescentOptimizer(0.01)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    return apply_gradient_op 


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-100-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)   