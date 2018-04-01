import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

from ops import *
from utils import *

slim = tf.contrib.slim

class GAN(object):
    def __init__(self, config):
        self.devices = config.devices
        self.config = config
        self.learning_rate = config.learning_rate

        self.generator = NetworkWrapper(self, config.generator_func)
        self.discriminator = NetworkWrapper(self, config.discriminator_func)

        #self.evaluate = Evaluate(self, config.eval_func)

        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.image_shape = config.image_shape
        self.sample_dir = config.sample_dir
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.y_dim = config.y_dim
        self.c_dim = config.c_dim
        self.f_dim = config.f_dim
        self.fc_dim = config.fc_dim
        self.z_dim = config.z_dim

        self.dataset_name = config.dataset
        self.dataset_path = config.dataset_path
        self.checkpoint_dir = config.checkpoint_dir

        self.use_augmentation = config.use_augmentation
        # self.is_training = tf.get_variable(
            # 'is_training', initializer=tf.constant_initializer(True), trainable=False)

    def save(self, sess, checkpoint_dir, step):
        model_name = "model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        # model_dir = "%s_%s" % (self.batch_size, self.learning_rate)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #print os.path.join(checkpoint_dir, ckpt_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print "Bad checkpoint: ", ckpt
            return False

    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith('d_')]
        self.g_vars = [var for var in t_vars if var.name.startswith('g_')]

        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars

    def build_model(self, is_training, learning_rate=None, global_step=0):
        config = self.config
        if learning_rate is not None:
            self.learning_rate = learning_rate

        # input
        self.image = tf.placeholder(tf.float32, shape=[self.batch_size]+self.image_shape)
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        image = preprocess_image(self.image, self.dataset_name, self.use_augmentation)

        #self.z = make_z(shape=[self.batch_size, self.z_dim])

        d_out_real = self.discriminator(image, is_training)

        self.gen_image = self.generator(self.z, is_training)
        d_out_fake = self.discriminator(self.gen_image, is_training, reuse=True)

        d_loss, g_loss, self.d_real, self.d_fake = self.get_loss(d_out_real, d_out_fake, config.loss)

        # optimizer
        self.get_vars()
        d_opt = tf.train.AdamOptimizer(
            self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        g_opt = tf.train.AdamOptimizer(
            self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        d_optimize = slim.learning.create_train_op(
            d_loss, d_opt, global_step=global_step, variables_to_train=self.d_vars)
        g_optimize = slim.learning.create_train_op(
            g_loss, g_opt, global_step=global_step, variables_to_train=self.g_vars)

        # logging
        tf.summary.scalar("d_real", self.d_real)
        tf.summary.scalar("d_fake", self.d_fake)
        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("g_loss", g_loss)
        tf.summary.image("fake_images", batch_to_grid(self.gen_image))
        # tf.summary.image("real_images", batch_to_grid(image))
        self.saver = tf.train.Saver(max_to_keep=None)

        self.eval_d = tf.sigmoid(self.discriminator(image, reuse=True))

        return d_optimize, g_optimize

    def get_loss(self, d_out_real, d_out_fake, loss='jsd'):
        sigm_ce = tf.nn.sigmoid_cross_entropy_with_logits
        loss_real = tf.reduce_mean(sigm_ce(logits=d_out_real, labels=tf.ones_like(d_out_real)))
        loss_fake = tf.reduce_mean(sigm_ce(logits=d_out_fake, labels=tf.zeros_like(d_out_fake)))
        loss_fake_ = tf.reduce_mean(sigm_ce(logits=d_out_fake, labels=tf.ones_like(d_out_fake)))

        if loss == 'jsd':
            d_loss = loss_real + loss_fake
            g_loss = - loss_fake
        elif loss == 'alternative':
            d_loss = loss_real + loss_fake
            g_loss = loss_fake_
        elif loss == 'reverse_kl':
            d_loss = loss_real + loss_fake
            g_loss = loss_fake_ - loss_fake

        return d_loss, g_loss, tf.reduce_mean(tf.nn.sigmoid(d_out_real)), tf.reduce_mean(tf.nn.sigmoid(d_out_fake))


class NetworkWrapper(object):
    def __init__(self, model, func):
        self.model = model
        self.func = func

    def __call__(self, z, is_training, reuse=False):
        return self.func(self.model, z, is_training, reuse=reuse)


