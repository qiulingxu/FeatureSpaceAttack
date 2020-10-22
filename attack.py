# Train the Style Transfer Net
from __future__ import print_function

import numpy as np
import sys
import os
import argparse
from PIL import Image

import tensorflow as tf

import settings
import dataprep
import modelprep

from style_transfer_net import StyleTransferNet_adv
from utils import  get_scope_var, save_rgb_img


np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(
    description='Training Auto Encoder for Feature Space Attack')
parser.add_argument("--dataset", help="Dataset for training the auto encoder",
                    choices=["imagenet", "cifar10"] , default="imagenet")
parser.add_argument("--decoder", help="Depth of the decoder to use. The deeper one (e.g. 3) injects more structure change. " +
                    "And it becomes more harmful but less nature-looking.", type=int, choices=[1, 2, 3], default=1)
parser.add_argument(
    "--scale", help="Whether to scale up the image size of CIFAR10 to the size of Imagenet", action="store_true")
parser.add_argument("--model", help="Model to attack.", default="imagenet_normal",
                    choices=["imagenet_normal", "imagenet_denoise", "cifar10_adv", "cifar10_nat", "cifar10_trades"])
parser.add_argument("--bound", help="Bound for attack, the exponential of sigma described in the paper", type=float, default=1.5)

args = parser.parse_args()

data_set = args.dataset
decoder = args.decoder
model_name = args.model
bound = args.bound

if data_set == "imagenet":
    decoder_list = {1: "imagenet_shallowest",
                    2: "imagenet_shallow",
                    3: "imagenet"}
    
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

task_name = "attack"

# Put all the pre-defined const into settings and fetch them as global variables
settings.common_const_init(data_set, model_name, decoder_name, task_name)
logger = settings.logger

for k, v in settings.config.items():
    globals()[k] = v

dataprep.init_data("eval")
get_data = dataprep.get_data


# (height, width, color_channels)
TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]

EPOCHS = 4
EPSILON = 1e-5
BATCH_SIZE = settings.config["BATCH_SIZE"]
if data_set == "cifar10":
    LEARNING_RATE = 1e-2
    LR_DECAY_RATE = 1e-4
    DECAY_STEPS = 1.0
    adv_weight = 500
    ITER=2000
    CLIP_NORM_VALUE = 10.0
else:
    if model_name .find("shallowest")>=0:
        LEARNING_RATE = 5e-3
    else:
        LEARNING_RATE = 1e-2
    LR_DECAY_RATE = 1e-3 
    DECAY_STEPS = 1.0
    adv_weight = 128 
    ITER=500
    CLIP_NORM_VALUE = 10.0

style_weight = 1


encoder_path = ENCODER_WEIGHTS_PATH
debug = True
if debug:
    from datetime import datetime
    start_time = datetime.now()

def grad_attack():
    sess.run(stn.init_style, feed_dict=fdict)
    sess.run(global_step.initializer)
    rst_img, rst_loss, nat_acc, rst_acc,rst_mean,rst_sigma = sess.run(
        [adv_img, content_loss_y, nat_output.acc_y_auto, adv_output.acc_y_auto, stn.meanS, stn.sigmaS],  feed_dict=fdict)
    print("Nature Acc:", nat_acc)
    for i in range(ITER):
        # Run an optimization step
        _ = sess.run([train_op],  feed_dict=fdict)
        
        # Clip the bound
        sess.run(stn.style_bound, feed_dict = fdict)
        
        # Monitor the progress
        _adv_img, acc, aloss, closs, _mean, _sigma = sess.run(
            [adv_img, adv_output.acc_y_auto, adv_loss, content_loss_y, stn.meanS, stn.sigmaS],  feed_dict=fdict)
        for j in range(BATCH_SIZE):
            # Save the best samples
            if acc[j]<rst_acc[j] or (acc[j]==rst_acc[j] and closs[j]<rst_loss[j]): 
                rst_img[j]=_adv_img[j]
                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_mean[j] = _mean[j]
                rst_sigma[j] = _sigma[j]

        if i%50==0 :
            acc=np.mean(acc)
            print(i,acc,"advl",aloss,"contentl",closs)
    
    # Reload the best saved samples
    sess.run(stn.asgn, feed_dict={stn.meanS_ph: rst_mean, stn.sigmaS_ph: rst_sigma})
    return rst_img


# get the traing image shape
HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
INPUT_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

# Gradient Clip in case of numerical instability
def gradient(opt, vars, loss ):
    gradients, variables = zip(*opt.compute_gradients(loss,vars))
    g_split = [tf.unstack(g, BATCH_SIZE, axis=0) for g in gradients]
    g1_list=[]
    g2_list=[]
    DIM = settings.config["DECODER_DIM"][-1]
    limit = 10/np.sqrt(DIM)    
    for g1,g2 in zip(g_split[0],g_split[1]):
        #(g1, g2), _ = tf.clip_by_global_norm([g1, g2], CLIP_NORM_VALUE)
        g1 = tf.clip_by_value(g1,-1/np.sqrt(limit),1/np.sqrt(limit))
        g2 = tf.clip_by_value(g2,-1/np.sqrt(limit),1/np.sqrt(limit))
        g1_list.append(g1)
        g2_list.append(g2)
    gradients = [tf.stack(g1_list, axis=0), tf.stack(g2_list, axis=0)]
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
    return opt


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

    content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
    label = tf.placeholder(tf.int64, shape=None, name="label")

    # create the style transfer net
    stn = StyleTransferNet_adv(encoder_path)

    # pass content and style to the stn, getting the generated_img
    dec_img, adv_img = stn.transform(content, p=bound)
    img = content

    stn_vars = get_scope_var("transform")
    # get the target feature maps which is the output of AdaIN
    target_features = stn.target_features

    # pass the generated_img to the encoder, and use the output compute loss
    enc_gen_adv, enc_gen_layers_adv = stn.encode(adv_img)
    
    modelprep.init_classifier()
    build_model = modelprep.build_model
    restore_model = modelprep.restore_model

    # Get the output from different input, this is a class which define different properties derived from logits
    # To use your own model, you can get your own logits from content and pass it to class build_logits in utils.py
    adv_output = build_model(adv_img, label, reuse=False)
    nat_output = build_model(img, label, reuse=True)
    dec_output = build_model(dec_img, label, reuse=True)

    # We are minimizing the loss. Take the negative of the loss
    # Use CW loss top5 for imagenet and CW top1 for cifar10. 
    # Here the target_loss represents CW loss, it is not the loss for targeted attack.
    adv_loss = -adv_output.target_loss_auto

    # compute the content loss
    content_loss_y = tf.reduce_sum(
        tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]),axis=-1)
    content_loss = tf.reduce_sum(content_loss_y)

    # compute the total loss
    loss = content_loss + tf.reduce_sum(adv_loss * BATCH_SIZE * adv_weight)
    decoder_vars = get_scope_var("decoder")

    # Training step
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)

    train_op = gradient(tf.train.AdamOptimizer(learning_rate, beta1= 0.5),vars=stn_vars, loss=loss)

    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(decoder_vars, max_to_keep=1)
    saver.restore(sess,Decoder_Model)
    restore_model(sess)
    ###### Start Training ######
    step = 0

    if debug:
        elapsed_time = datetime.now() - start_time
        start_time = datetime.now()
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        print('Now begin to train the model...\n')

    uid = 0

    report_batch = 50
    for batch in range(1,100+1):

        if batch % report_batch == 1:
            np_adv_image = []
            np_benign_image = []
            np_content_loss = []
            np_acc_attack = []
            np_acc_attack_5 = []
            np_acc = []
            np_acc_5 = []
            np_decode_acc = []
            np_decode_acc_5 = []
            np_acc_5 = []
            np_label = []
        # run the training step
        
        x_batch, y_batch = get_data()
        fdict = {content: x_batch, label: y_batch}
        grad_attack()

        step += 1

        for i in range(BATCH_SIZE):
            gan_out = sess.run(adv_img, feed_dict=fdict)
            save_out = np.concatenate(
                (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
            sz = TRAINING_IMAGE_SHAPE[1]
            full_path = os.path.join(
                base_dir_model, "%d" % step,  "%d.jpg" % i)
            os.makedirs(os.path.join(base_dir_model, "%d" %
                                     step), exist_ok=True)
            save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
            save_rgb_img(save_out, path=full_path)

        if batch % 1 == 0:

            elapsed_time = datetime.now() - start_time
            _content_loss, _adv_acc, _adv_loss, _loss,   \
                = sess.run([content_loss, adv_output.accuracy, adv_loss, loss, ], feed_dict=fdict)
            _adv_img, _loss_y, _adv_acc_y, _adv_acc_y_5, _acc_y, _acc_y_5, _decode_acc_y, _decode_acc_y_5 = sess.run([
                adv_img, content_loss_y, adv_output.acc_y, adv_output.acc_y_5, nat_output.acc_y, nat_output.acc_y_5, dec_output.acc_y, dec_output.acc_y_5], feed_dict=fdict)

            np_adv_image.append(_adv_img)
            np_benign_image.append(x_batch)
            np_content_loss.append(_loss_y)
            np_acc_attack.append(_adv_acc_y)
            np_acc_attack_5 .append(_adv_acc_y_5)
            np_acc_5 .append(_acc_y_5)
            np_acc .append(_acc_y)
            np_label.append(y_batch)
            np_decode_acc.append(_decode_acc_y)
            np_decode_acc_5.append(_decode_acc_y_5)

            _adv_loss = np.sum(_adv_loss)

            diff = (_adv_img - x_batch) /255
            l2_norm = np.sum(diff*diff)
            li_norm = np.mean( np.amax(np.abs(diff), axis=-1))
            l1_norm = np.mean(np.sum(np.abs(diff), axis=-1))

            print("l2_norm", l2_norm, "li_norm", li_norm, "l1_loss", l1_norm)
            print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
            print('content loss: %.3f' % (_content_loss))
            print('adv loss  : %.3f,  weighted adv loss: %.3f , adv acc %.3f' %
                  (_adv_loss, adv_weight * _adv_loss, _adv_acc))
            print("normal acc:", _acc_y)
            print("adv acc:", _adv_acc_y)
            print("normal acc top5:", _acc_y_5)
            print("adv acc top5:", _adv_acc_y_5)


        if batch % report_batch == 0:
            np_adv_image_arr = np.concatenate(np_adv_image)
            np_benign_image_arr = np.concatenate(np_benign_image)
            np_content_loss_arr = np.concatenate(np_content_loss)
            np_acc_attack_arr = np.concatenate(np_acc_attack)
            np_acc_attack_5_arr = np.concatenate(np_acc_attack_5)
            np_acc_arr = np.concatenate(np_acc)
            np_acc_5_arr = np.concatenate(np_acc_5)
            np_decode_acc_arr = np.concatenate(np_decode_acc)
            np_decode_acc_5_arr = np.concatenate(np_decode_acc_5)
            np_label_arr = np.concatenate(np_label)

            saved_dict = {"adv_image": np_adv_image_arr, 
                        "benign_image": np_benign_image_arr,
                        "content_loss": np_content_loss_arr,
                        "acc_attack": np_acc_attack_arr,
                        "acc_attack_5": np_acc_attack_5_arr,
                        "acc": np_acc_arr,
                        "acc_5": np_acc_5_arr,   
                        "decode_acc": np_decode_acc_arr,
                        "decode_acc_5": np_decode_acc_5_arr,
                          "label": np_label_arr}

            np.save(os.path.join(base_dir_model, "saved_samples%d.npy" %
                                 (batch//report_batch)), saved_dict)

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

