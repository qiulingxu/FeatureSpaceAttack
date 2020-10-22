# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import logging
from PIL import Image

import settings
import dataprep
import modelprep

from style_transfer_net import StyleTransferNet
from adaptive_instance_norm import normalize
from utils import save_rgb_img, get_scope_var

import argparse

parser = argparse.ArgumentParser(description='Training Auto Encoder for Feature Space Attack')
parser.add_argument("--dataset", help="Dataset for training the auto encoder", choices=["imagenet", "cifar10"])
parser.add_argument("--decoder", help="Depth of the decoder to use. The deeper one injects more structure change. " +\
    "And it becomes more harmful but less nature-looking.", type=int, choices=[1,2,3], default=1)
parser.add_argument("--scale", help="Whether to scale up the image size of CIFAR10 to the size of Imagenet. " +\
    "Scaling up image size provides better adversarial samples, but consumes larger memory.", action="store_true")
args = parser.parse_args()

data_set = args.dataset
decoder = args.decoder
if data_set == "imagenet":
    decoder_list = {1: "imagenet_shallowest", 
                    2: "imagenet_shallow",
                    3: "imagenet"}
    model_name = "imagenet_normal"
    decoder_name = decoder_list[decoder]

elif data_set == "cifar10":
    # One can choose to not to scale CIFAR10 to Imagenet for better speed. While for best quality, one need to consider scale the image size up 
    # The corresponding decoder name is cifar10_unscale
    decoder_list = {1: "cifar10_shallowest",
                    2: "cifar10_shallow",
                    3: "cifar10"}
    if args.scale:
        decoder_name = decoder_list[decoder]
    else:
        decoder_name = "cifar10_unscale"
    model_name = "cifar10_nat"
    
task_name = "train"

# Put all the pre-defined const into settings and fetch them as global variables
settings.common_const_init(data_set,model_name,decoder_name,task_name)
logger=settings.logger

for k, v in settings.config.items():
    globals()[k] = v

dataprep.init_data("train")
get_data = dataprep.get_data
get_data_pair = dataprep.get_data_pair

TRAINING_IMAGE_SHAPE = IMAGE_SHAPE#settings.config["IMAGE_SHAPE"]


LEARNING_RATE = 1e-4
LR_DECAY_RATE = 2e-5
EPSILON = 1e-5
# 2e-5  30000 -> half
DECAY_STEPS = 1.0
adv_weight = 500
style_weight = settings.config["style_weight"]


get_data()
encoder_path = 'vgg19_normalised.npz'

debug=True
logging_period=100
if debug:
    from datetime import datetime
    start_time = datetime.now()

# get the traing image shape
HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
INPUT_SHAPE = [None, HEIGHT, WIDTH, CHANNELS]
# create the graph
tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
tf_config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

    content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
    style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
    label = tf.placeholder(tf.int64, shape =None, name="label")
    #style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')

    # create the style transfer net
    stn = StyleTransferNet(encoder_path)

    # pass content and style to the stn, getting the gen_img
    # decoded image from normal one, adversarial image, and input
    dec_img, adv_img = stn.transform(content, style)
    img = content

    print(adv_img.shape.as_list())
    stn_vars = []

    # get the target feature maps which is the output of AdaIN
    target_features = stn.target_features

    # pass the gen_img to the encoder, and use the output compute loss
    enc_gen_adv, enc_gen_layers_adv = stn.encode(adv_img)
    enc_gen, enc_gen_layers = stn.encode(dec_img)

    l2_embed = normalize(enc_gen)[0] - normalize(stn.norm_features)[0]
    l2_embed = tf.reduce_mean(tf.sqrt(tf.reduce_sum((l2_embed * l2_embed),axis=[1,2,3])))

    # compute the content loss
    content_loss = tf.reduce_sum(tf.reduce_mean(
        tf.square(enc_gen_adv - target_features), axis=[1, 2])) 

    modelprep.init_classifier()
    build_model = modelprep.build_model
    restore_model = modelprep.restore_model

    # Get the output from different input, this is a class which define different properties derived from logits
    # To use your own model, you can get your own logits from content and pass it to class build_logits in utils.py
    adv_output = build_model(adv_img, label, reuse=False)
    nat_output = build_model(img, label, reuse=True)
    dec_output = build_model(dec_img, label, reuse=True)
    
    style_layer_loss = []
    for layer in STYLE_LAYERS:
        enc_style_feat = stn.encoded_style_layers[layer]
        enc_gen_feat = enc_gen_layers_adv[layer]  

        meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
        meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

        sigmaS = tf.sqrt(varS + EPSILON)
        sigmaG = tf.sqrt(varG + EPSILON)

        l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
        l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

        style_layer_loss.append(l2_mean + l2_sigma)

    style_loss = tf.reduce_sum(style_layer_loss)

    # compute the total loss

    loss = content_loss + style_weight * style_loss 

    decoder_vars = get_scope_var("decoder")
    # Training step
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=stn_vars+decoder_vars, global_step=global_step)  # stn_vars+

    sess.run(tf.global_variables_initializer())
    restore_model(sess)

    # saver
    saver = tf.train.Saver(stn_vars+decoder_vars, max_to_keep=1)
    step = 0

    if debug:
        elapsed_time = datetime.now() - start_time
        start_time = datetime.now()
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        print('Now begin to train the model...\n')


    # For imagenet, it takes around 2 days for training
    for batch in range(300000):

        # run the training step
        x_batch, y_batch, x_batch_style, y_batch_style = get_data_pair()
        fdict = {content: x_batch, label: y_batch, style: x_batch_style}

        
        if step % 1000 == 0:
            saver.save(sess, model_save_path, global_step=step, write_meta_graph=False)

        if batch % 1000 ==0:
            img_path = os.path.join(decoder_name + "img%.2f" % style_weight,  "%d" % step)
            for i in range(8):
                gan_out = sess.run(adv_img, feed_dict=fdict)
                save_out = np.concatenate(
                    (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
                full_path = os.path.join(img_path, "%d.jpg" % i)
                os.makedirs(img_path, exist_ok=True)
                sz=TRAINING_IMAGE_SHAPE[1]
                save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
                save_rgb_img(save_out, path=full_path)

        if batch % 100 == 0:

            elapsed_time = datetime.now() - start_time
            _content_loss, _adv_acc, _adv_loss, _loss, _l2_embed = sess.run([content_loss, adv_output.acc, adv_output.target_loss, loss, l2_embed],
                                                                  feed_dict=fdict)
            _normal_loss, _normal_acc = sess.run([nat_output.target_loss, nat_output.acc],
                                                 feed_dict=fdict)

            logger.info('step: %d,  total loss: %.3f,  elapsed time: %s' %
                   (step, _loss, elapsed_time))
            logger.info('content loss: %.3f' % (_content_loss))
            logger.info('adv loss  : %.3f,  weighted adv loss: %.3f , adv acc %.3f' %
                  (_adv_loss, adv_weight * _adv_loss, _adv_acc))
            logger.info('normal loss : %.3f normal acc: %.3f l2_embed %.3f\n' %
                  (_normal_loss, _normal_acc, _l2_embed))

        sess.run(train_op, feed_dict=fdict)
        step += 1

    ###### Done Training & Save the model ######
    saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        print('Model is saved to: %s' % model_save_path)

