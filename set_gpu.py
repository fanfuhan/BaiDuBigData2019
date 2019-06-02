import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def set_gpu():
    # 指定使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"

    # keras 设置gpu内存按需分配
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
