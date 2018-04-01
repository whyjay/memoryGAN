import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

from models.gan import *
from models.memory import *
from ops import *
from utils import *

slim = tf.contrib.slim

class MemoryGAN(GAN):
    def __init__(self, config):
        super(MemoryGAN, self).__init__(config)

        self.lamb = config.lamb
        self.key_dim = config.key_dim
        self.mem_size = config.mem_size
        self.mem = BaseMemory(self.key_dim, self.mem_size, choose_k=config.choose_k)

    def build_model(self, is_training, learning_rate=None, global_step=0):
        config = self.config
        if learning_rate is not None:
            self.learning_rate = learning_rate

        # input
        self.image = tf.placeholder(tf.float32, shape=[self.batch_size]+self.image_shape)
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        image = preprocess_image(self.image, self.dataset_name, self.use_augmentation)

        q_r = self.discriminator(image, is_training)
        d_out_r = self.mem.query(q_r, tf.ones(self.batch_size))

        self.q_sample = self.mem.sample_histogram(self.batch_size)
        self.gen_image = self.generator(
            tf.concat([self.z, self.q_sample], axis=1), is_training)

        q_f = self.discriminator(self.gen_image, is_training, reuse=True)
        d_out_f = self.mem.query(q_f, tf.zeros(self.batch_size))

        d_loss, g_loss, self.d_real, self.d_fake = self.get_loss(d_out_r, d_out_f, config.loss)
        mi_loss = tf.losses.mean_squared_error(q_f, self.q_sample)
        d_loss += self.lamb * mi_loss
        g_loss += self.lamb * mi_loss

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

        # # logging
        tf.summary.scalar("d_real", self.d_real)
        tf.summary.scalar("d_fake", self.d_fake)
        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("g_loss", g_loss)
        tf.summary.scalar("mem_real_rate", tf.reduce_mean(self.mem.mem_vals))
        tf.summary.image("fake_images", batch_to_grid(self.gen_image))
        tf.summary.image("real_images", batch_to_grid(image))
        self.saver = tf.train.Saver(max_to_keep=None)

        return d_optimize, g_optimize

    def get_loss(self, d_out_real, d_out_fake, loss='jsd'):
        def cross_entropy(y, smooth=1e-3):
            y = tf.minimum(1-smooth, tf.maximum(smooth, y))
            return tf.reduce_mean(-tf.log(y))

        loss_real = cross_entropy(d_out_real)
        loss_fake = cross_entropy(1 - d_out_fake)
        loss_fake_ = cross_entropy(d_out_fake)

        if loss == 'jsd':
            d_loss = loss_real + loss_fake
            g_loss = - loss_fake
        elif loss == 'alternative':
            d_loss = loss_real + loss_fake
            g_loss = loss_fake_
        elif loss == 'reverse_kl':
            d_loss = loss_real + loss_fake
            g_loss = loss_fake_ - loss_fake
        elif loss == 'updown':
            d_loss = tf.reduce_mean(- d_out_real + d_out_fake)
            g_loss = tf.reduce_mean(- d_out_fake)

        return d_loss, g_loss, tf.reduce_mean(d_out_real), tf.reduce_mean(d_out_fake)

