import tensorflow as tf
import os
import time
import numpy as np
from utils import *
from ops import *

def test_generation(model, sess):
    config = model.config
    bs = config.batch_size
    n_batches = 10000
    n_split = 1
    filesize = bs*n_batches/n_split

    global_step = tf.Variable(0, trainable=False)
    is_training = False
    d_optim, g_optim = model.build_model(is_training, global_step=global_step)
    res = model.load(sess, config.load_cp_dir)
    print res

    print "[*] Test Start"
    start_time = time.time()

    samples = generate_memgan(model, sess, n_iter=n_batches)

    total_time = time.time() - start_time
    print "[*] Finished : %f" % (total_time)

    image_dir = os.path.join('samples', config.load_cp_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i in range(n_split):
        with open(os.path.join(image_dir, 'samples_gen_%d.npy' % i), 'w') as f:
            np.save(f, samples[i*filesize:(i+1)*filesize])

    print "saved at %s" % image_dir
    sess.close()

def generate_memgan(model, sess, n_iter=100):
    samples = []

    for i in range(n_iter):
        sample = sess.run(model.gen_image, feed_dict={model.z:get_z(model)})
        samples.append(sample)

    return np.concatenate(samples, axis=0)

def get_z(model):
    return np.random.uniform(-1., 1., size=(model.batch_size, model.z_dim))

