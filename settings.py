import tensorflow as tf
import logging
import sys
import os

def init_settings(config_name,suffix="",task_dir=""):
    global config
    config={}
    assert config_name in ["cifar10", "cifar10_shallo", "cifar10_shallowest", "cifar10_unscale", "imagenet",
                        "imagenet_shallow", "imagenet_shallowest"]
    if config_name.find("cifar10")>=0:
        data_set = "cifar10"
    elif config_name.find("imagenet") >= 0:
        data_set = "imagenet"
    else:
        assert False

    config["config_name"] = config_name
    config["data_set"] = data_set
    config["style_weight"]=1
    
    # data mode:
    # 1: allowing any pairs to feed into training
    # 2: only allowing pairs from the same class to feed into training
    config["data_mode"] = 2
    config["INTERPOLATE_NUM"] = 50 + 1

    if config_name.find("_shallowest")>=0:
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1')
        config["DECODER_LAYERS"] = ('conv2_1', 'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, )
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1')
    elif config_name.find("_shallow")>=0:
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', )
        config["DECODER_LAYERS"] = ('conv3_1',
                                    'conv2_2', 'conv2_1',
                                    'conv1_2', 'conv1_1')
        config["upsample_indices"] = (1, 3)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1')
    else:
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1')
        config["DECODER_LAYERS"] = ('conv4_1',
                                    'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1',
                                    'conv2_2', 'conv2_1',
                                    'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, 4, 6)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

    if config_name == "cifar10_unscale":
        config["CLASS_NUM"] = 10
        config["IMAGE_SHAPE"] = [32,32,3]
        config["DECODER_DIM"] = [16, 16, 128]

        config["NO_SCALE"] = True
        config["BATCH_SIZE"] = 64
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1')
        config["DECODER_LAYERS"] = ('conv2_1', 'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, )
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1')

        config["pretrained_model"] = "pretrained.ckpt"
        config["hardened_model"] = "hardened.ckpt"
        config["model_save_path"] = "./cifar10transform%d.ckpt" % (config["style_weight"])

    if config_name == "cifar10_shallowest":
        config["CLASS_NUM"] = 10
        config["IMAGE_SHAPE"] = [32, 32, 3]
        config["DECODER_DIM"] = [112, 112, 128]

        config["pretrained_model"] = "pretrained.ckpt"
        config["hardened_model"] = "hardened.ckpt"
        config["model_save_path"] = "./cifar10shallowesttransform_scale%d.ckpt" % (
            config["style_weight"])

    elif config_name == "cifar10_shallow":
        config["CLASS_NUM"] = 10
        config["IMAGE_SHAPE"] = [32, 32, 3]
        config["DECODER_DIM"] = [112, 112, 128]

        config["pretrained_model"] = "pretrained.ckpt"
        config["hardened_model"] = "hardened.ckpt"
        config["model_save_path"] = "./cifar10shallowtransform_scale%d.ckpt" % (
            config["style_weight"])
        
    elif config_name == "cifar10":
        config["CLASS_NUM"] = 10
        config["IMAGE_SHAPE"] = [32, 32, 3]
        config["DECODER_DIM"] = [28, 28, 512]

        config["pretrained_model"] = "pretrained.ckpt"
        config["hardened_model"] = "hardened.ckpt"
        config["model_save_path"] = "./cifar10transform_scale%d.ckpt" % (
            config["style_weight"])

    elif config_name == "imagenet":
        config["CLASS_NUM"] = 1000
        
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["DECODER_DIM"] = [28, 28, 512]

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenettransform%d.ckpt" % (
            config["style_weight"])


    elif config_name == "imagenet_shallow":
        config["CLASS_NUM"] = 1000
        config["INTERPOLATE_NUM"] = 50 + 1
        config["DECODER_DIM"] = [56, 56, 256]
        config["IMAGE_SHAPE"] = [224, 224, 3]


        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenetshallowtransform%d.ckpt" % (
            config["style_weight"])

    elif config_name == "imagenet_shallowest":
        config["CLASS_NUM"] = 1000
        config["INTERPOLATE_NUM"] = 50 + 1
        config["DECODER_DIM"] = [112, 112, 128]
        config["IMAGE_SHAPE"] = [224, 224, 3]

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenetshallowesttransform%d.ckpt" % (
            config["style_weight"])

    global logger

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT,
                        filename=task_dir+"log.log")
    logger = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def common_const_init(data_set, model_name, decoder_name, task_name):
    
    global config


    assert data_set in ["imagenet", "cifar10"]
    ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
    base_dir_data = os.path.join("store", data_set)
    base_dir_decoder = os.path.join("store", data_set, decoder_name)
    base_dir_model = os.path.join("store", data_set, decoder_name, model_name)
    task_dir = os.path.join("store", data_set, decoder_name, model_name, task_name)
    os.makedirs(task_dir, exist_ok=True)

    if data_set == "cifar10":
        assert model_name in ["cifar10_nat", "cifar10_adv", "cifar10_trades"]
        assert decoder_name in ["cifar10", "cifar10_balance"]
        init_settings(decoder_name, task_dir=task_dir)

        Decoder_Model = "./cifar10transform1.ckpt"

        

    elif data_set == "imagenet":
        assert model_name in ["imagenet_denoise", "imagenet_normal"]
        assert decoder_name in ["imagenet",
                                "imagenet_shallow", "imagenet_shallowest"]
        init_settings(decoder_name, task_dir=task_dir)
        from imagenetmod.interface import imagenet

        if decoder_name == "imagenet_shallowest":
            Decoder_Model = "./imagenetshallowesttransform1.ckpt.mode2"
        elif decoder_name == "imagenet_shallow":
            # "./trans_pretrained/imagenetshallowtransform1.ckpt-104000"
            Decoder_Model = "./imagenetshallowtransform1.ckpt.mode2"
        elif decoder_name == "imagenet":
            Decoder_Model = "./imagenettransform1.ckpt.mode2"

    print(locals())
    config.update(locals())
