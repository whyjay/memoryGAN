import os
import numpy as np
import tensorflow as tf

from models.config import Config
from models.memory_gan import MemoryGAN
from models.test_generation import test_generation
from models.train import train
from utils import pp, visualize, to_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1500, "Max epoch to train")
flags.DEFINE_string("exp", 0, "Experiment number")
flags.DEFINE_string("load_cp_dir", '', "cp path")
flags.DEFINE_string("dataset", "fashion", "[fashion, affmnist, cifar10]")
flags.DEFINE_string("loss", "jsd", "[jsd, alternative, reverse_kl, updown]")
flags.DEFINE_boolean("lr_decay", False, "")
flags.DEFINE_boolean("use_augmentation", False, "")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_string("model", 'MemoryGAN', '')
flags.DEFINE_string("generator", 'base_g', '')
flags.DEFINE_string("discriminator", 'memory_d', '')

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    config = Config(FLAGS)
    config.print_config()
    config.make_dirs()

    config_proto = tf.ConfigProto(allow_soft_placement=FLAGS.is_train, log_device_placement=False)
    config_proto.gpu_options.allow_growth = True

    with tf.Session(config=config_proto) as sess:
        model = globals()[FLAGS.model](config)

        if not FLAGS.is_train:
            test_generation(model, sess)
        else:
            train(model, sess)

if __name__ == '__main__':
    tf.app.run()
