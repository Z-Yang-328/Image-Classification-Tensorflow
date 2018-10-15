
## The build_network.py provides steps to build a complete convolutional neural network.
## Each step is written as a function.

import tensorflow as tf

# Build the network

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    input_image = tf.placeholder(tf.float32,
                                 shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name='x')

    return input_image


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    input_labels = tf.placeholder(tf.float32,
                                  shape=[None, n_classes],
                                  name='y')

    return input_labels


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    dropout_prob = tf.placeholder(tf.float32, name='keep_prob')

    return dropout_prob


# Convolution and Max Pooling Layer

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    shape = shape = (conv_ksize[0],
                     conv_ksize[1],
                     x_tensor.get_shape().as_list()[-1],
                     conv_num_outputs)
    weight = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    conv_layer = tf.nn.conv2d(x_tensor,
                              weight,
                              strides=[1, conv_strides[0], conv_strides[1], 1],
                              padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.max_pool(conv_layer,
                                ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                                strides=[1, pool_strides[0], pool_strides[1], 1],
                                padding='SAME')

    return conv_layer


# Flatten Layer

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    image_shape = x_tensor.get_shape().as_list()
    flattened = image_shape[1] * image_shape[2] * image_shape[3]

    x_tensor_reshaped = tf.reshape(x_tensor, shape=(-1, flattened))

    return x_tensor_reshaped


# Fully-Connected Layer

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = (x_tensor.get_shape().as_list()[-1], num_outputs)
    weight = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))
    biases = tf.Variable(tf.zeros(num_outputs))
    x_fully = tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), biases))

    return x_fully


# Output Layer

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = (x_tensor.get_shape().as_list()[-1], num_outputs)
    weight = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=0.1))
    biases = tf.Variable(tf.zeros(num_outputs))
    output = tf.add(tf.matmul(x_tensor, weight), biases)

    return output


# Create Convolutional Model

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Convolutional parameters
    conv_num_outputs = 18
    conv_ksize = (4, 4)
    conv_strides = (1, 1)
    pool_ksize = (8, 8)
    pool_strides = (1, 1)
    num_outputs_fully = 200
    num_outputs_out = 10

    # Convolutional layer
    conv1 = conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    # Dropout layer to avoid overfitting
    conv1_dropout = tf.nn.dropout(conv1, keep_prob)

    # Flatten the image
    flattened = flatten(conv1_dropout)

    # Fully connect layer
    fconn1 = fully_conn(flattened, num_outputs_fully)

    # Output logits
    out_logits = output(fconn1, num_outputs_out)

    return out_logits