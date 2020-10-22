# Encoder is fixed to the first few layers (up to relu4_1)
# of VGG-19 (pre-trained on ImageNet)
# This code is a modified version of Anish Athalye's vgg.py
# https://github.com/anishathalye/neural-style/blob/master/vgg.py

import numpy as np
import tensorflow as tf
import settings


class Encoder(object):

    def __init__(self, weights_path):
        # load weights (kernel and bias) from npz file
        weights = np.load(weights_path)

        idx = 0
        self.weight_vars = []
        ENCODER_LAYERS = settings.config["ENCODER_LAYERS"]
        # create the TensorFlow variables
        with tf.variable_scope('encoder'):
            for layer in ENCODER_LAYERS:
                kind = layer[:4]

                if kind == 'conv':
                    kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
                    bias   = weights['arr_%d' % (idx + 1)]
                    kernel = kernel.astype(np.float32)
                    bias   = bias.astype(np.float32)
                    idx += 2

                    with tf.variable_scope(layer):
                        W = tf.Variable(kernel, trainable=False, name='kernel')
                        b = tf.Variable(bias,   trainable=False, name='bias')

                    self.weight_vars.append((W, b))

    def encode(self, image):

        # create the computational graph
        idx = 0
        layers = {}
        current = image
        ENCODER_LAYERS = settings.config["ENCODER_LAYERS"]
        for i,layer in enumerate(ENCODER_LAYERS):
            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                current = conv2d(current, kernel, bias)

            elif kind == 'relu':
                current = tf.nn.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            layers[layer] = current
        print("encoder %d shape: " % i, current.shape.as_list())

        assert(len(layers) == len(ENCODER_LAYERS))

        enc = layers[ENCODER_LAYERS[-1]]

        return enc, layers

    def preprocess(self, image, mode='RGB'):
        assert mode == "RGB"
        # preprocess
        if settings.config["IMAGE_SHAPE"][0] != 224 and "NO_SCALE" not in settings.config:
            if "pre_scale" in image.__dict__:
                image = image.pre_scale
            else:
                image = tf.image.resize(image, size=[224, 224])
        
        # To BGR
        image = tf.reverse(image, axis=[-1])
        
        return image - np.array([103.939, 116.779, 123.68])

    def deprocess(self, image, mode='BGR'):
        assert mode == "BGR"
        image =  image + np.array([103.939, 116.779, 123.68])
        
        image = tf.reverse(image, axis=[-1])
        image = tf.clip_by_value(image, 0.0, 255.0)
        
        pre_scale = image
        if settings.config["IMAGE_SHAPE"][0] != 224 and "NO_SCALE" not in settings.config:
            image = tf.image.resize(
                image, size=settings.config["IMAGE_SHAPE"][:2])
        image.pre_scale = pre_scale
        return image

def conv2d(x, kernel, bias):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

