import tensorflow as tf
import numpy as np
import settings
from PIL import Image

def save_rgb_img(img, path):
    img = img.astype(np.uint8)
    #img=np.reshape(img,[28,28])
    Image.fromarray(img, mode='RGB').save(path)


def get_scope_var(scope_name, only_train = False):
    if only_train:
        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    else:
        var_list = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    assert (len(var_list) >= 1)
    return var_list


def get_shape(x):
    x_shape = x.get_shape().as_list()
    if x_shape[0] is None:
        return [tf.shape(x)[0]]+x_shape[1:]
    else:
        return x_shape


# Copyright @ https://jhui.github.io/2017/03/07/TensorFlow-GPU/
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


confidence = 50
l2_weight = 1e-5

def top_k_acc(logits,labels,k):
    return tf.cast(tf.nn.in_top_k(predictions=logits,
                           targets=labels, k=5), tf.float32)

class build_logits:

    def __init__(self, logits, label, conf=1):
        self._build(logits, label, conf)

    def _build(self, logits, label, conf):
        classes = logits.shape.as_list()[1]
        self.logits = logits
        self.labels = label
        self.onehot_label = tf.one_hot(label, depth=classes)
        self.prediction = tf.argmax(logits, axis=-1)
        self.acc_y = tf.cast(tf.equal(self.prediction, label), tf.float32)
        self.acc = tf.reduce_mean(self.acc_y)
        self.logits = logits
        self.label_logit = tf.reduce_sum(self.onehot_label*logits)
        self.wrong_logit = tf.reduce_max(
            (1-self.onehot_label)*logits - self.onehot_label*1e9)
        self.target_loss = tf.nn.relu(
            self.label_logit-self.wrong_logit+confidence)
        self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logits)
        self.xent_sum = tf.reduce_sum(self.xent)
        self.xent_mean = tf.reduce_mean(self.xent)

        self.acc_y_5 = top_k_acc(self.logits, self.labels, k=5 )
        self.acc_5 = tf.reduce_mean(self.acc_y_5)
        
        self.wrong_logit5, _idx = tf.nn.top_k(
            logits * (1-self.onehot_label) - self.onehot_label * 1e7, k=5, sorted=False)
        self.true_logit5 = tf.reduce_sum(
            logits * self.onehot_label, axis=-1, keep_dims=True)

        # The higher, the more successful of adv attack
        self.target_loss5 = - \
            tf.reduce_sum(tf.nn.relu(self.true_logit5 - self.wrong_logit5 + conf), axis=1)
        if classes>50:
            self.accuracy = self.acc_5
            self.target_loss_auto = self.target_loss5
        else:
            self.accuracy = self.acc
            self.target_loss_auto = self.target_loss

def normalize(content,  epsilon=1e-5):
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    #meanC_s, varC_s = tf.nn.moments(content, [1, 2])
    bs = settings.config["BATCH_SIZE"]
    content_shape = content.shape.as_list()
    new_shape = [bs, 1, 1, content_shape[3]]

    sigmaC = tf.sqrt(tf.add(varC, epsilon))
    #sigmaS = tf.sqrt(tf.add(varS, epsilon))
    normalize_content = (content - meanC) / sigmaC

    return normalize_content, meanC, sigmaC

