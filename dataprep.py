import functools
import numpy as np

import settings

from imagenetmod.interface import imagenet
from cifar10 import cifar10_input

class datapairs():
    def __init__(self, class_num, batch_size, stack_num=10):
        self.class_num = class_num
        self.batch_size = batch_size
        self.bucket = [[] for _ in range(self.class_num)]
        self.bucket_size = [0 for _ in range(self.class_num)]
        self.tot_pair = 0
        self.index = 0
        self.stack_num = stack_num
        self.loaded = 0

    def add_data(self, x, y):

        if self.bucket_size[y] < self.stack_num:
            self.bucket[y].append(x)
            self.bucket_size[y] += 1
            if self.bucket_size[y] == self.stack_num:
                self.loaded += 1
                self.bucket[y] = np.stack(self.bucket[y])

    def feed_pair(self, x_batch, y_batch):
        for i in range(self.batch_size):
            self.add_data(x_batch[i], y_batch[i])
        if self.loaded == self.class_num:
            return False
        else:
            return True

class datapair():
    def __init__(self,class_num, batch_size):
        self.class_num=class_num
        self.batch_size=batch_size
        self.bucket=[ [] for _ in range(self.class_num)]
        self.bucket_size= [0 for _ in range(self.class_num)]
        self.tot_pair=0
        self.index=0

    def add_data(self,x,y):
        self.bucket_size[y]+=1
        
        if self.bucket_size[y] % 2==0:
            self.tot_pair+=1
        self.bucket[y].append(x)

    def feed_pair(self,x_batch,y_batch):
        for i in range(self.batch_size):
            self.add_data(x_batch[i],y_batch[i])
    
    def get_pair(self):
        if self.tot_pair<self.batch_size:
            return None
        else:
            x1=[]
            y1=[]
            x2=[]
            y2=[]
            left=self.batch_size
            i = self.index  # ensure random start of each class
            for _ in range(self.class_num):
                if left==0:
                    break
                sz = self.bucket_size[i]
                if sz>=2:
                    pairs=min(left,sz//2)
                else:
                    i = (i+1) % self.class_num
                    continue
                x1.extend(self.bucket[i][:pairs])
                x2.extend(self.bucket[i][pairs:2*pairs])
                y1.extend([i]*pairs)
                y2.extend([i]*pairs)
                self.bucket[i] = self.bucket[i][2*pairs:]
                self.bucket_size[i]-=2*pairs
                left-=pairs
                i= (i+1)%self.class_num
                #print(i)
            self.index = i
            self.tot_pair-=self.batch_size
            x1=np.stack(x1)
            x2=np.stack(x2)
            y1=np.stack(y1)
            y2=np.stack(y2)
        return x1,y1,x2,y2


def init_data(mode):
    global CLASS_NUM, BATCH_SIZE, inet, cifar_data, data_set, dp, config_name, raw_cifar
    assert mode in ["train","eval"]
    CLASS_NUM = settings.config["CLASS_NUM"]
    BATCH_SIZE = settings.config["BATCH_SIZE"]
    data_set = settings.config["data_set"]
    config_name = settings.config["config_name"]

    assert data_set in ["cifar10","svhn","imagenet"]
    data_set = data_set

    if data_set == "imagenet":
        if mode == "train":
            inet = imagenet(BATCH_SIZE, dataset="train")
        elif mode == "eval":
            inet = imagenet(BATCH_SIZE, dataset="val")
    elif data_set == "cifar10":

        raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")
        if mode == "eval":
            cifar_data = raw_cifar.eval_data
        elif mode == "train":
            cifar_data = raw_cifar.train_data
    else:
        assert False, "Not implemented"
    dp = datapair(CLASS_NUM, BATCH_SIZE)

def init_polygon_data(stack_num, fetch_embed):
    global _mean_all, _sigma_all 
    mean_file = "polygon_mean_%s.npy" % config_name
    sigma_file = "polygon_sigma_%s.npy" % config_name
    if os.path.exists(mean_file) and os.path.exists(sigma_file):
        _mean_all = np.load(mean_file)
        _sigma_all = np.load(sigma_file)
    else:
        ## Populate polygon point
        dps = datapairs(CLASS_NUM, BATCH_SIZE, stack_num)
        f = True
        while f:
            x_batch, y_batch = get_data()
            f = dp.feed_pair(x_batch, y_batch)
        print("datapairs loading")
        polygon_arr = np.concatenate(dp.bucket)
        len_arr = polygon_arr.shape[0]
        _mean = []
        _sigma = []
        for i in range((len_arr - 1) // BATCH_SIZE + 1):
            # sess.run([stn.meanC, stn.sigmaC], feed_dict={
            _meanC, _sigmaC = fetch_embed(
                polygon_arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            _mean.append(_meanC)
            _sigma.append(_sigmaC)
        print("datapairs loaded")
        _mean_all = np.concatenate(_mean, axis=0)
        _sigma_all = np.concatenate(_sigma, axis=0)
        np.save(mean_file, _mean_all)
        np.save(sigma_file, _sigma_all)

def popoulate_data(_meanC, _sigmaC, y_batch, include_self=True):

    res_mean = []
    res_sigma = []

    if include_self:
        real_num = INTERPOLATE_NUM - 1
        for i in range(BATCH_SIZE):
            y = y_batch[i]
            meanCi = _meanC[i: i+1]
            meanC_pop = _mean_all[y*real_num:(y+1)*real_num]
            res_mean.append(np.concatenate([meanCi, meanC_pop]))
            sigmaCi = _sigmaC[i: i+1]
            sigmaC_pop = _sigma_all[y*real_num:(y+1)*real_num]
            res_sigma.append(np.concatenate([sigmaCi, sigmaC_pop]))
    else:
        real_num = INTERPOLATE_NUM
        for i in range(BATCH_SIZE):
            y = y_batch[i]
            meanC_pop = _mean_all[y*real_num:(y+1)*real_num]
            res_mean.append(meanC_pop)
            sigmaCi = _sigmaC[i: i+1]
            sigmaC_pop = _sigma_all[y*real_num:(y+1)*real_num]
            res_sigma.append(sigmaC_pop)
    return np.stack(res_mean), np.stack(res_sigma)

def get_fetch_func(sess, content, pred):
    return functools.partial(_fetch_embed, sess=sess, content=content, pred=pred)

def _fetch_embed(sess, content, pred ): 
    _pred = sess.run(pred, feed_dict={content: content})
    return _pred

def _get_data():

    if data_set =="cifar10":
        x_batch, y_batch = cifar_data.get_next_batch(
                batch_size=BATCH_SIZE, multiple_passes=True)
    elif data_set == "imagenet":
         x_batch, y_batch = inet.get_next_batch()
    
    return x_batch,y_batch

def get_data():
    return _get_data()

def get_data_pair():
    mode = settings.config["data_mode"]
    if mode == 1:
        ret_list = []
        for _ in range(2):
            ret_list.extend(get_data())
        return ret_list

    else:
        res = dp.get_pair()
        while res is None:
            x_batch, y_batch = get_data()
            dp.feed_pair(x_batch, y_batch)
            res = dp.get_pair()
        return res
