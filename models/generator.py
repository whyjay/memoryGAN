import tensorflow as tf
from ops import *

def base_g(model, z, is_training, reuse=False):
    bs = model.batch_size
    f_dim = model.f_dim
    fc_dim = model.fc_dim
    c_dim = model.c_dim

    with slim.arg_scope(ops_with_bn, is_training=is_training):
        with tf.variable_scope('g_', reuse=reuse) as scope:

            if model.dataset_name == 'fashion':
                n_layer = 2
                w = model.image_shape[0]

                h = fc(z, fc_dim, act=lrelu)
                h = fc(h, f_dim*2*w/4*w/4)
                h = tf.reshape(h, [-1, w/4, w/4, f_dim*2])
                h = deconv2d(h, f_dim*2, 4, 2)
                x = deconv2d(h, c_dim, 4, 2, act=tf.nn.sigmoid, norm=None)

            elif model.dataset_name == 'affmnist':
                n_layer = 3
                c = 2**(n_layer - 1)
                w = model.image_shape[0]/2**(n_layer)

                h = fc(z, f_dim * c * w * w, act=lrelu)
                h = tf.reshape(h, [-1, w, w, f_dim * c])

                for i in range(n_layer - 1):
                    w *= 2
                    c /= 2
                    h = deconv2d(h, f_dim * c, 4, 2)
                    h = deconv2d(h, f_dim * c, 1, 1)

                x = deconv2d(h, c_dim, 4, 2, act=tf.nn.sigmoid, norm=None)

            elif model.dataset_name == 'cifar10':
                n_layer = 3
                w = model.image_shape[0]/2**(n_layer)

                h = fc(z, f_dim * w * w, act=tf.nn.elu, norm=ln)
                h = tf.reshape(h, [-1, w, w, f_dim])

                c = f_dim
                for i in range(n_layer):
                    c /= 2
                    h = residual_block(h, resample='up', act=tf.nn.elu, norm=ln)

                x = conv2d(h, c_dim, 3, 1, act=tf.nn.tanh, norm=None)
    return x

