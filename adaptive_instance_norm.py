# Adaptive Instance Normalization

import tensorflow as tf
import settings

def AdaIN(content, style, epsilon=1e-5):
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    meanS, varS = tf.nn.moments(style,   [1, 2], keep_dims=True)

    sigmaC = tf.sqrt(tf.add(varC, epsilon))
    sigmaS = tf.sqrt(tf.add(varS, epsilon))

    return (content - meanC) * sigmaS / sigmaC + meanS, meanS, sigmaS

def AdaIN_adv_tanh(content,  epsilon=1e-5):
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    bs = settings.config["BATCH_SIZE"]
    content_shape = content.shape.as_list()
    new_shape = [bs, 1, 1, content_shape[3]]
    with tf.variable_scope("scale"):
        sigmaS = tf.get_variable("sigma_S", shape=new_shape,
                            initializer=tf.zeros_initializer())
        meanS = tf.get_variable("mean_S", shape=new_shape,
                                  initializer=tf.zeros_initializer())
        

    sigmaC = tf.sqrt(tf.add(varC, epsilon))


    p=tf.sqrt(1.5)

    def get_mid_range(l,r):
        _mid=(l+r)/2.0
        _range=(r-l)/2.0
        return _mid,_range

    sign=tf.sign(meanC)
    abs_meanC=tf.abs(meanC)

    _sigma_mid, _sigma_range = get_mid_range(sigmaC/p, sigmaC*p)
    _mean_mid, _mean_range = get_mid_range(abs_meanC/p, abs_meanC*p)

    sigmaSp = _sigma_range*tf.nn.tanh(sigmaS)+_sigma_mid
    meanSp = sign * (_mean_range*tf.nn.tanh(meanS)+_mean_mid)

    ops_bound = []

    ops_asgn = [tf.assign(sigmaS, tf.atanh((sigmaC-_sigma_mid)/ (_sigma_range +1e-4) )), 
                tf.assign(meanS, tf.atanh((abs_meanC-_mean_mid)/(_mean_range + 1e-4) ))]

    #ops_asgn = [sigmaS.initializer, meanS.initializer]#
    #ops_asgn = [tf.assign(sigmaS, sigmaC-_sigma_mid),
    #            tf.assign(meanS, meanC-_mean_mid)]

    return (content - meanC) * sigmaSp / sigmaC + meanSp , ops_asgn, ops_bound, sigmaSp, meanSp, meanS, sigmaS


def AdaIN_adv(content,  epsilon=1e-5, p=1.5):
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    bs = settings.config["BATCH_SIZE"]
    content_shape = content.shape.as_list()
    new_shape = [bs, 1, 1, content_shape[3]]
    with tf.variable_scope("scale"):
        meanS = tf.get_variable("mean_S", shape=new_shape,
                                initializer=tf.zeros_initializer())
        sigmaS = tf.get_variable("sigma_S", shape=new_shape,
                                 initializer=tf.ones_initializer())


    sigmaC = tf.sqrt(tf.add(varC, epsilon))

    #p = 1.5
    p_sigma = p
    p_mean = p

    sign = tf.sign(meanC)
    abs_meanC = tf.abs(meanC)
    ops_bound = [tf.assign(sigmaS, tf.clip_by_value(sigmaS, sigmaC/p_sigma, sigmaC*p_sigma)),
                 tf.assign(meanS, tf.clip_by_value(meanS, abs_meanC/p_mean, abs_meanC*p_mean))]

    sigmaC_rand = tf.random_uniform(tf.shape(sigmaC), sigmaC/p, sigmaC*p)
    meanC_rand = tf.random_uniform(tf.shape(meanC), abs_meanC/p, abs_meanC*p)
    #sigmaS = tf.sqrt(tf.add(varS, epsilon))
    ops_asgn = [tf.assign(meanS, abs_meanC), tf.assign(sigmaS, sigmaC)]
    ops_asgn_rand = [tf.assign(sigmaS, sigmaC_rand), tf.assign(meanS, meanC_rand)]

    return (content - meanC) * sigmaS / sigmaC + sign * meanS, ops_asgn, ops_bound, sigmaS, meanS, meanC, sigmaC, ops_asgn_rand, (content - meanC) / (sigmaC)


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
