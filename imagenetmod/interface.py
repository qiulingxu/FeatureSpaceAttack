import tensorflow as tf
from tensorpack.tfutils import SmartInit
from .nets import ResNeXtDenoiseAllModel
from .third_party.imagenet_utils import get_val_dataflow
from tensorpack.tfutils.tower import TowerContext

def restore_parameter(sess):
    file_path = "X101-DenoiseAll.npz"
    sessinit = SmartInit(file_path)
    sessinit.init(sess)

class container:
    def __init__(self):
        pass


def build_imagenet_model(image, label, reuse=False, conf=1):
    args = container()
    args.depth = 101
    with TowerContext(tower_name='', is_training=False):
        with tf.variable_scope("", auxiliary_name_scope=False, reuse=reuse):
            model = ResNeXtDenoiseAllModel(args)
            model.build_graph(image, label)
    return model.logits

def build_imagenet_model_old(image,label,reuse=False,conf =1):
    args=container()
    args.depth=101
    with TowerContext(tower_name='', is_training=False):
        with tf.variable_scope("", auxiliary_name_scope=False, reuse=reuse):
            model=ResNeXtDenoiseAllModel(args)
            model.build_graph(image,label)
    cont = container
    cont.logits = model.logits
    cont.label = tf.argmax(cont.logits, axis=-1)
    cont.acc_y = 1-model.wrong_1
    cont.acc_y_5 = 1-model.wrong_5
    cont.accuracy = tf.reduce_mean(1-model.wrong_1) # wrong_5
    cont.rev_xent = tf.reduce_mean(tf.log(
        1 - tf.reduce_sum(tf.nn.softmax(model.logits) *
                          tf.one_hot(label, depth=1000), axis=-1) 
    ))
    cont.poss_loss = 1 - tf.reduce_mean(
        tf.reduce_sum(tf.nn.softmax(model.logits) *
                          tf.one_hot(label, depth=1000), axis=-1)
    )

    label_one_hot = tf.one_hot(label, depth=1000)
    wrong_logit = tf.reduce_max(model.logits * (1-label_one_hot) -label_one_hot * 1e7, axis=-1)
    true_logit = tf.reduce_sum(model.logits * label_one_hot, axis=-1)
    #wrong_logit = tf.contrib.nn.nth_element(model.logits * (1-label_one_hot) - label_one_hot * 1e7, n=5, reverse=True)
    wrong_logit5, _idx = tf.nn.top_k(
        model.logits * (1-label_one_hot) - label_one_hot * 1e7, k=5, sorted=False)
    true_logit5 = tf.reduce_sum(model.logits * label_one_hot, axis=-1, keep_dims=True)
    cont.target_loss5 = - tf.reduce_sum(tf.nn.relu(true_logit5 - wrong_logit5 + conf), axis=1)
    cont.target_loss = - tf.nn.relu(true_logit - wrong_logit + conf)
    cont.xent_filter = tf.reduce_mean((1.0-model.wrong_1)*
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=model.logits), axis=-1)

    cont.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=model.logits), axis=-1)
    #cont.target_loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    labels=label, logits=model.logits) * tf.nn.relu(tf.minimum(1.0, true_logit - wrong_logit + conf))
    return cont

class imagenet:

    def __init__(self, batchsize, dataset="val"):
        self.batchsize=batchsize
        self.dataset=dataset
        self.init()

    def init(self, ):
        self.data = get_val_dataflow(
            "imagenet", self.batchsize, dataname=self.dataset)
        self.data.reset_state()
        self.iter=iter(self.data)
        #self.data = tf.transpose(data, [0, 3, 1, 2])


    def get_next_batch(self):
        pack=next(self.iter,None)
        if pack is None:
            self.data.reset_state()
            self.iter = iter(self.data)
            pack = next(self.iter, None)
        return pack
