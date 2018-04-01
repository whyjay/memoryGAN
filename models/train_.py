import tensorflow as tf
import os
import time
import numpy as np
from utils import *
from ops import *
from IPython import embed

def train(model, sess):
    config = model.config
    dataset = load_dataset(model)
    N = dataset.num_examples
    max_iter = int(N/model.batch_size) * config.epoch

    global_step = tf.Variable(0, trainable=False)
    if model.lr_decay:
        learning_rate = tf.train.polynomial_decay(
            model.learning_rate, global_step, max_iter, 0.)
    else:
        learning_rate = None

    d_optim, g_optim = model.build_model(learning_rate, global_step)
    if not (config.load_cp_dir == ''):
        model.load(sess, config.load_cp_dir)
    coord, threads, merged_sum = init_training(model, sess)
    start_time = time.time()
    print_time = time.time()


    print "[*] Traing Start : N=%d, Batch=%d, epoch=%d, max_iter=%d" \
        %(N, model.batch_size, config.epoch, max_iter)

    try:
        for idx in xrange(1, max_iter):
            batch_start_time = time.time()

            # D step
            image, label = dataset.next_batch(model.batch_size)
            _, d_real, d_fake = sess.run(
                [d_optim, model.d_real, model.d_fake],
                feed_dict={model.image:image, model.label:label, model.z:get_z(model)})
            '''
            # Wasserstein
            _ = sess.run([model.clip_d_op])
            '''

            # G step
            _ = sess.run([g_optim], feed_dict={model.image:image, model.label:label, model.z:get_z(model)})

            # save checkpoint for every epoch
            if ((idx*model.batch_size) % N < model.batch_size) and idx > 1:
                epoch = int(idx*model.batch_size/N) + 1
                print_time = time.time()
                total_time = print_time - start_time
                sec_per_epoch = (print_time - start_time) / epoch

                # image, label = dataset.next_batch(model.batch_size)
                # summary = sess.run([merged_sum], feed_dict={model.image:image, model.label:label, model.z:get_z(model)})[0]
                # model.writer.add_summary(summary, epoch)

                if epoch > 100:
                    _save_samples(model, sess, epoch)
                    model.save(sess, model.checkpoint_dir, epoch)

                print '[Epoch %(epoch)d] time: %(total_time)4.4f, d_real: %(d_real).8f, d_fake: %(d_fake).8f, sec_per_epoch: %(sec_per_epoch)4.4f' % locals()

    except tf.errors.OutOfRangeError:
        print "Done training; epoch limit reached."
    # finally:
        # coord.request_stop()
    # coord.join(threads)
    sess.close()

def _save_samples(model, sess, epoch):
    samples = []
    noises = []

    # generator hard codes the batch size
    for i in xrange(model.sample_size // model.batch_size):
        # gen_image, noise = sess.run([model.gen_image, model.z])
        gen_image, noise = sess.run([model.gen_image, model.z],
                                    feed_dict={model.z:get_z(model)})
        samples.append(gen_image)
        noises.append(noise)

    samples = np.concatenate(samples, axis=0)
    noises = np.concatenate(noises, axis=0)

    assert samples.shape[0] == model.sample_size
    save_images(samples, [8, 8], os.path.join(model.sample_dir, 'samples_%s.png' % (epoch)))

    print  "Save Samples at %s/%s" % (model.sample_dir, 'samples_%s' % (epoch))
    with open(os.path.join(model.sample_dir, 'samples_%d.npy'%(epoch)), 'w') as f:
        np.save(f, samples)
    with open(os.path.join(model.sample_dir, 'noises_%d.npy'%(epoch)), 'w') as f:
        np.save(f, noises)

def init_training(model, sess):
    config = model.config
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    merged_sum = tf.summary.merge_all()
    model.writer = tf.summary.FileWriter(config.log_dir, sess.graph)

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    coord = None
    threads = None

    if model.load(sess, model.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    if not os.path.exists(config.dataset_path):
        print(" [!] Data does not exist : %s" % config.dataset_path)
    return coord, threads, merged_sum

def load_dataset(model):
    if model.dataset_name == 'mnist':
        import mnist as ds
    elif model.dataset_name == 'fashion':
        import fashion as ds
    elif model.dataset_name == 'affmnist':
        import affmnist as ds
    elif model.dataset_name == 'cifar10':
        import cifar10 as ds
    elif model.dataset_name == 'celeba':
        import celeba as ds
    elif model.dataset_name == 'lsun':
        import lsun as ds
    elif model.dataset_name == 'chair':
        import chair as ds
    return ds.read_data_sets(model.dataset_path, dtype=tf.uint8, reshape=False, validation_size=0).train


def get_z(model):
    return np.random.uniform(-1., 1., size=(model.batch_size, model.z_dim))
    # z =  np.random.normal(0., model.stddev, size=(model.batch_size, model.key_dim))
    # return z/np.linalg.norm(z)
