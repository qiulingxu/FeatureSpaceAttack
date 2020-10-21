import tensorflow as tf
from . import resnet_slim
slim = tf.contrib.slim

def get_scope_var(scope_name):
    var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    assert (len(var_list) >= 1)
    return var_list

def restore_parameter(sess):
    file_path = "imagenet_resnet_v1_50.ckpt"
    var_list = get_scope_var("resnet_v1")
    saver = tf.train.Saver(var_list)
    saver.restore(sess,file_path)



class container:
    def __init__(self):
        pass

def compute_loss_and_error(logits, label, label_smoothing=0.):
    if label_smoothing == 0.:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label)
    else:
        nclass = logits.shape[-1]
        loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(label, nclass),
            logits, label_smoothing=label_smoothing,
            reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(loss, name='xentropy-loss')

    def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name=name)

    wrong_1 = prediction_incorrect(logits, label, 1, name='wrong-top1')

    wrong_5 = prediction_incorrect(logits, label, 5, name='wrong-top5')
    return loss, wrong_1, wrong_5

def build_imagenet_model(image, label, reuse=False, conf=1, shrink_class = 1000):

    with slim.arg_scope(resnet_slim.resnet_arg_scope()):
        logits, desc = resnet_slim.resnet_v1_50(image, num_classes=shrink_class, is_training= False, reuse=reuse)
    return logits

