import os
import tensorflow as tf
from net_config.mbnv2_384x576_32x3_300_imgnet import MyNet
from utills.base import save_model

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

net = MyNet()
def resume_from_checkpoint(net):
    net.init_model()
    latest = tf.train.latest_checkpoint("checkpoints/"+net.name+"/")
    print(latest)
    net.model.load_weights(latest)
    return net.model
    save_model(model_name=net.name, model=net.model)


