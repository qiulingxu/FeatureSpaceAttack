import tensorflow as tf
import numpy as np

import settings
import utils
import dataprep

import imagenetmod.interface as imagenet_denoise_interface
import models.pretrained.interface as imagenet_normal_interface
from models import cifar10_class as resnet_cifar10 
from models import trade_interface as cifar_wrn_trades_interface


def init_classifier(conf = 1):
    global build_model, restore_model
    model_name=settings.config["model_name"]
    assert model_name in ["imagenet_denoise", "imagenet_normal", "cifar10_nat", "cifar10_adv", "cifar10_trades"]
    if model_name in ["imagenet_denoise"]:
        
        def _build_model(input,label,reuse):
            input = tf.reverse(input, axis=[-1]) # rgb to bgr
            logits = imagenet_denoise_interface.build_imagenet_model(
                input, label, reuse, conf=conf)
            container = utils.build_logits (logits, label, conf)
            return container

        _restore_model = imagenet_denoise_interface.restore_parameter

    elif model_name in ["imagenet_normal"]:
        def _build_model(input, label, reuse):
            # refer to https://github.com/tensorflow/models/blob/6e63dfee4118df6e889227b1a32badf7d0a09e3b/research/slim/preprocessing/vgg_preprocessing.py
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _mean = np.array([_R_MEAN, _G_MEAN, _B_MEAN]).reshape([1,1,1,-1])
            input = input - _mean

            logits = imagenet_normal_interface.build_imagenet_model(
                input, label, reuse, conf=conf)
            container = utils.build_logits(logits, label, conf)
            return container

        _restore_model = imagenet_normal_interface.restore_parameter

    elif model_name in ["cifar10_nat","cifar10_adv"]:
        def _build_model(input, label, reuse):
            model = resnet_cifar10.Model("eval", dataprep.raw_cifar.train_images)
            model._build_model(input, label, reuse, conf = conf)
            container = utils.build_logits(model.logits, label, conf)
            return container

        def _restore_model(sess):
            classifier_vars = utils.get_scope_var("model")
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            if model_name == "cifar10_nat": 
                classifier_saver.restore(sess, "./pretrained/pretrained.ckpt")
            elif model_name == "cifar10_adv":
                classifier_saver.restore(sess, "./pretrained/hardened.ckpt")

    elif model_name in ["cifar10_trades"]:
        def _build_model(input, label, reuse):
            assert settings.config["BATCH_SIZE"] == 64 , "Graph is static and the batch size must be 64"
            logits = cifar_wrn_trades_interface.get_model(input)
            container = utils.build_logits(logits, label, conf)
            return container

        def _restore_model(sess):
            pass

    restore_model = _restore_model
    build_model = _build_model
