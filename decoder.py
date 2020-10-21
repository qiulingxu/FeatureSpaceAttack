# Decoder mostly mirrors the encoder with all pooling layers replaced by nearest
# up-sampling to reduce checker-board effects.
# Decoder has no BN/IN layers.

import tensorflow as tf
import settings

class Decoder(object):

    def __init__(self):
        self.weight_vars = []

        with tf.variable_scope('decoder'):
            self._create_variables(512, 256, 3, scope='conv4_1')

            self._create_variables(256, 256, 3, scope='conv3_4')
            self._create_variables(256, 256, 3, scope='conv3_3')
            self._create_variables(256, 256, 3, scope='conv3_2')
            self._create_variables(256, 128, 3, scope='conv3_1')

            self._create_variables(128, 128, 3, scope='conv2_2')
            self._create_variables(128,  64, 3, scope='conv2_1')

            self._create_variables( 64,  64, 3, scope='conv1_2')
            self._create_variables( 64,   3, 3, scope='conv1_1')

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        if scope in settings.config["DECODER_LAYERS"]:

            with tf.variable_scope(scope):
                shape  = [kernel_size, kernel_size, input_filters, output_filters]
                kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=shape, name='kernel')
                bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=[output_filters], name='bias')
                pack = (kernel, bias)
            self.weight_vars.append(pack)

    def _create_variables_t(self, input_filters, output_filters, kernel_size, scope):
        assert False
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(
                uniform=False), shape=shape, name='kernel')
            bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(
                uniform=False), shape=[input_filters], name='bias')
            return (kernel, bias)

    def decode(self, image):
        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = settings.config["upsample_indices"]
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            #print("decoder in %d shape: " % i, out.shape.as_list())
            kernel, bias = self.weight_vars[i]
            #if i in upsample_indices:
            #    out=transconv2d(out,kernel,bias)
            #else:
            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            
            if i in upsample_indices:
                out = upsample(out)
            #print("decoder out %d shape: "%i, out.shape.as_list())

        return out


def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out


def transconv2d(x, kernel, bias, use_relu=True):


    bs=tf.shape(x)[0]
    img_sz=x.shape.as_list()[1]
    #print(img_sz)
    filter_size=kernel.shape.as_list()[2]
    # conv and add bias
    g_deconv = tf.nn.conv2d_transpose(x, kernel, output_shape=[
        bs, img_sz*2, img_sz*2, filter_size], strides=[1, 2, 2, 1], padding='SAME')
    out = g_deconv + bias

    if use_relu:
        out = tf.nn.relu(out)
    #print(out.shape.as_list())
    return out


def upsample(x, scale=2):
    height = x.shape.as_list()[1]*scale#tf.shape(x)[1] * scale
    width = x.shape.as_list()[2]*scale  # tf.shape(x)[2] * scale
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return output

