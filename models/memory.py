import numpy as np
import tensorflow as tf


class BaseMemory(object):
    def __init__(self, key_dim, memory_size, choose_k=256,
                 correct_in_top=1, var_cache_device='', nn_device=''):
        self.beta=1e-8
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.choose_k = min(choose_k, memory_size)
        self.correct_in_top = correct_in_top
        self.var_cache_device = var_cache_device  # Variables are cached here.
        self.nn_device = nn_device  # Device to perform nearest neighbour matmul.

        caching_device = var_cache_device if var_cache_device else None
        self.mem_keys = tf.get_variable(
            'memkeys', [self.memory_size, self.key_dim], trainable=False,
            initializer=tf.random_uniform_initializer(-0.0, 0.0),
            caching_device=caching_device)

        mem_val_init = np.ones(self.memory_size, dtype=np.float32)
        mem_val_init[self.memory_size/2:] = 0.
        self.mem_vals = tf.get_variable(
            'memvals', [self.memory_size], trainable=False,
            initializer=tf.constant_initializer(mem_val_init),
            caching_device=caching_device)

        self.mem_age = tf.get_variable(
            'memage', [self.memory_size], dtype=tf.float32, trainable=False,
            initializer=tf.constant_initializer(0.0), caching_device=caching_device)

        self.mem_hist = tf.get_variable(
            name="memory_hist", shape=[self.memory_size], trainable=False,
            initializer=tf.constant_initializer(1e-5, dtype=np.float32))

    def get(self):
        return self.mem_keys, self.mem_vals, self.mem_age, self.mem_hist

    def set(self, k, v, a, h, r=None):
        return tf.group(
            self.mem_keys.assign(k),
            self.mem_vals.assign(v),
            self.mem_age.assign(a),
            self.mem_hist.assign(h))

    def clear(self):
        return tf.variables_initializer(
            [self.mem_keys, self.mem_vals, self.mem_age, self.mem_hist])

    def get_hint_pool_idxs(self, normalized_query, label=None):
        assert normalized_query.get_shape().as_list()[1] == self.key_dim
        with tf.device(self.nn_device):
            similarities = tf.matmul(tf.stop_gradient(normalized_query),
                                     self.mem_keys, transpose_b=True, name='nn_mmul')

        if not (None == label):
            is_wrong = tf.to_float(
                tf.abs(tf.expand_dims(label, 1) - tf.expand_dims(self.mem_vals, axis=0)))
            is_wrong = tf.minimum(1.0, is_wrong)
            similarities = similarities - 2*is_wrong

        likelihood = tf.exp(similarities-1.)
        prior = tf.stop_gradient(self.mem_hist + self.beta)
        _, k_idxs = tf.nn.top_k(
            tf.stop_gradient(prior*likelihood), k=self.choose_k, name='nn_topk')
        self.lld = tf.log(tf.reduce_sum(1./np.sqrt(2*np.pi)*tf.exp(similarities-1.)*(self.mem_hist*self.mem_vals + self.beta)/tf.reduce_sum(self.mem_hist*self.mem_vals + self.beta), axis=1))
        return k_idxs

    def make_update_op(self, upd_idxs, upd_keys, upd_vals, upd_hists, batch_size):
        mem_age_incr = self.mem_age.assign_add(tf.ones([self.memory_size],
                                                    dtype=tf.float32))
        with tf.control_dependencies([mem_age_incr]):
            mem_age_upd = tf.scatter_update(
                self.mem_age, upd_idxs, tf.zeros([batch_size*self.choose_k], dtype=tf.float32))

        mem_key_upd = tf.scatter_update(
            self.mem_keys, upd_idxs, upd_keys)
        mem_val_upd = tf.scatter_update(
            self.mem_vals, upd_idxs, upd_vals)
        mem_hist_upd = tf.scatter_update(
            self.mem_hist, upd_idxs, upd_hists)

        return tf.group(mem_age_upd, mem_key_upd, mem_val_upd, mem_hist_upd)

    def get_histogram(self):
        return self.mem_hist/tf.reduce_sum(self.mem_hist)

    def sample_histogram(self, n, is_key=True):
        real_hist = self.mem_hist * self.mem_vals
        probs = real_hist/tf.reduce_sum(real_hist)
        self.probs = probs
        distr = tf.contrib.distributions.Categorical(probs=probs)
        idxs = distr.sample(n)
        if is_key:
            sample_keys = tf.reshape(tf.gather(self.mem_keys, idxs), [n, -1])
            return sample_keys
        return tf.one_hot(idxs, self.memory_size)

    def sample_onehot(self, n):
        real_hist = self.mem_hist * self.mem_vals
        probs = real_hist/tf.reduce_sum(real_hist)
        self.probs = probs
        distr = tf.contrib.distributions.Categorical(probs=probs)
        idxs = distr.sample(n)
        sample_keys = tf.reshape(tf.gather(self.mem_keys, idxs), [n, -1])
        return sample_keys

    def query(self, query_vec, label, update_memory=tf.constant(True),
            alpha=0.5, n_iter=1):
        batch_size = tf.shape(query_vec)[0]

        q = tf.nn.l2_normalize(query_vec, dim=1)
        result, joint_, vals_ = self.get_result(q, alpha)
        reset_mask = self.get_reset_mask(label, joint_, vals_)

        # tile for multiple update
        rep_reset_mask = tf.reshape(tf.tile(tf.reshape(
            reset_mask, (batch_size, 1)), [1, self.choose_k]), [-1])
        rep_oldest_idxs = tf.reshape(tf.tile(
            self.get_oldest_idxs(batch_size), [1, self.choose_k]), [-1])
        rep_q = tf.reshape(tf.tile(
            tf.expand_dims(q, axis=1), [1, self.choose_k, 1]), [-1, self.key_dim])
        rep_label = tf.reshape(tf.tile(
            tf.expand_dims(label, axis=1), [1, self.choose_k]), [-1])

        # EM update
        k_idxs = self.get_hint_pool_idxs(q, label=label)

        # create small memory and look up with gradients
        with tf.device(self.var_cache_device):
            keys_upd = tf.stop_gradient(tf.gather(self.mem_keys, k_idxs, name='my_mem_keys_gather'))
            hists_upd = alpha * tf.stop_gradient(tf.gather(
                self.mem_hist, k_idxs, name='hint_pool_mem_hist'))
            vals_upd = tf.gather(self.mem_vals, k_idxs, name='fetched_vals')

        prev_posterior = 0.

        for _ in range(n_iter):
            # E step
            similarities = tf.reduce_sum(tf.expand_dims(q, axis=1)*keys_upd, axis=2) # batch_size * choose_k
            likelihood = tf.exp(similarities-1.)
            prior = hists_upd + self.beta
            joint = likelihood * prior
            posterior = joint / tf.reduce_sum(joint, axis=1, keep_dims=True)
            hists_upd = hists_upd + tf.reduce_sum(posterior-prev_posterior, axis=1, keep_dims=True)
            upd_ratio = (posterior-prev_posterior)/hists_upd # batch_size X choose_k

            # M-step
            keys_upd = keys_upd*(1-tf.expand_dims(upd_ratio, axis=2))
            keys_upd += tf.expand_dims(q, axis=1)*tf.expand_dims(upd_ratio, axis=2)
            keys_upd = tf.nn.l2_normalize(keys_upd, dim=2)

            prev_posterior = posterior

        with tf.control_dependencies([result]):
            upd_idxs = tf.where(rep_reset_mask,
                                rep_oldest_idxs,
                                tf.reshape(k_idxs, [-1]))
            upd_keys = tf.where(rep_reset_mask,
                                rep_q,
                                tf.reshape(keys_upd, [-1, self.key_dim]))
            upd_vals = tf.where(rep_reset_mask,
                                rep_label,
                                tf.reshape(vals_upd, [-1]))
            upd_hists = tf.where(rep_reset_mask,
                                 tf.ones_like(rep_label)*tf.reduce_mean(self.mem_hist),
                                 tf.reshape(hists_upd, [-1]))

        def make_update_op():
            return self.make_update_op(upd_idxs, upd_keys, upd_vals, upd_hists, batch_size)

        update_op = tf.cond(update_memory, make_update_op, tf.no_op)

        with tf.control_dependencies([update_op]):
            result = tf.identity(result)

        return result

    def get_result(self, q, alpha):
        k_idxs = self.get_hint_pool_idxs(q)

        with tf.device(self.var_cache_device):
            keys = tf.stop_gradient(tf.gather(self.mem_keys, k_idxs, name='my_mem_keys_gather'))
            hists = tf.stop_gradient(tf.gather(self.mem_hist, k_idxs, name='hint_pool_mem_hist'))
            hists = hists * alpha
            vals = tf.gather(self.mem_vals, k_idxs, name='fetched_vals')

        similarities = tf.reduce_sum(tf.expand_dims(q, axis=1)*keys, axis=2)
        likelihood = tf.exp(similarities-1.)
        prior = hists + self.beta
        joint = likelihood * prior
        posterior = joint / tf.reduce_sum(joint, axis=1, keep_dims=True)
        result = tf.reduce_sum(posterior * vals, axis=1)
        return result, joint, vals

    def get_reset_mask(self, label, joint, vals):
        # get index of nearest correct answer
        teacher_hints = tf.to_float(tf.abs(tf.expand_dims(label, 1) - vals))
        teacher_hints = 1.0 - tf.minimum(1.0, teacher_hints)
        _, teacher_hint_idxs = tf.nn.top_k(joint * teacher_hints, k=1)
        sliced_hints = tf.slice(teacher_hints, [0, 0], [-1, self.correct_in_top])
        reset_mask = tf.equal(0.0, tf.reduce_sum(sliced_hints, 1))
        return reset_mask

    def get_oldest_idxs(self, batch_size):
        if self.choose_k < 64:
            _, oldest_idxs = tf.nn.top_k(self.mem_age, k=1, sorted=False)
            oldest_idxs = tf.tile(tf.reshape(oldest_idxs, (1, 1)), [batch_size, 1])
        else:
            _, oldest_idxs = tf.nn.top_k(self.mem_age, k=batch_size, sorted=False)
            oldest_idxs = tf.reshape(oldest_idxs, (batch_size, 1))
        return oldest_idxs
