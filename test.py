from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets import MnistDataset
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import gen_random_ops

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

root_log_dir = "logs/mnist"
root_checkpoint_dir = "ckt/mnist"
batch_size = 128
updates_per_epoch = 100
max_epoch = 50

exp_name = "mnist_%s" % timestamp

log_dir = os.path.join(root_log_dir, exp_name)
checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

mkdir_p(log_dir)
mkdir_p(checkpoint_dir)

dataset = MnistDataset()

latent_spec = [
    (Uniform(62), False),
    (Categorical(10), True),
    (Uniform(1, fix_std=True), True),
    (Uniform(1, fix_std=True), True),
]

output_dist=MeanBernoulli(dataset.image_dim),
image_shape=dataset.image_shape,
network_type="mnist",

model = RegularizedGAN(
    output_dist=MeanBernoulli(dataset.image_dim),
    latent_spec=latent_spec,
    batch_size=batch_size,
    image_shape=dataset.image_shape,
    network_type="mnist",
)

info_reg_coeff=1.0,
generator_learning_rate=1e-3,
discriminator_learning_rate=2e-4,

algo = InfoGANTrainer(
    model=model,
    dataset=dataset,
    batch_size=batch_size,
    exp_name=exp_name,
    log_dir=log_dir,
    checkpoint_dir=checkpoint_dir,
    max_epoch=max_epoch,
    updates_per_epoch=updates_per_epoch,
    info_reg_coeff=1.0,
    generator_learning_rate=1e-3,
    discriminator_learning_rate=2e-4,
)

from infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
import sys

TINY = 1e-8
REUSE = False

max_epoch=100,
updates_per_epoch=100,
snapshot_interval=1000, #initially it was 10000

def visualize_all_factors():
    with tf.Session():
        fixed_noncat = np.concatenate([
            np.tile(
                model.nonreg_latent_dist.sample_prior(10).eval(),
                [10, 1]
            ),
            model.nonreg_latent_dist.sample_prior(batch_size - 100).eval(),
        ], axis=0)
        fixed_cat = np.concatenate([
            np.tile(
                model.reg_latent_dist.sample_prior(10).eval(),
                [10, 1]
            ),
            model.reg_latent_dist.sample_prior(batch_size - 100).eval(),
        ], axis=0)
        #
    offset = 0
    for dist_idx, dist in enumerate(model.reg_latent_dist.dists):
        if isinstance(dist, Gaussian):
            assert dist.dim == 1, "Only dim=1 is currently supported"
            c_vals = []
            for idx in xrange(10):
                c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
            c_vals.extend([0.] * (batch_size - 100))
            vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
            cur_cat = np.copy(fixed_cat)
            cur_cat[:, offset:offset+1] = vary_cat
            offset += 1
        elif isinstance(dist, Categorical):
            lookup = np.eye(dist.dim, dtype=np.float32)
            cat_ids = []
            for idx in xrange(10):
                cat_ids.extend([idx] * 10)
            cat_ids.extend([0] * (batch_size - 100))
            cur_cat = np.copy(fixed_cat)
            cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
            offset += dist.dim
        elif isinstance(dist, Bernoulli):
            assert dist.dim == 1, "Only dim=1 is currently supported"
            lookup = np.eye(dist.dim, dtype=np.float32)
            cat_ids = []
            for idx in xrange(10):
                cat_ids.extend([int(idx / 5)] * 10)
            cat_ids.extend([0] * (batch_size - 100))
            cur_cat = np.copy(fixed_cat)
            cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
            # import ipdb; ipdb.set_trace()
            offset += dist.dim
        else:
            raise NotImplementedError
        z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))
        #
        _, x_dist_info = model.generate(z_var)
        #
        # just take the mean image
        if isinstance(model.output_dist, Bernoulli):
            img_var = x_dist_info["p"]
        elif isinstance(model.output_dist, Gaussian):
            img_var = x_dist_info["mean"]
        else:
            raise NotImplementedError
        img_var = dataset.inverse_transform(img_var)
        rows = 10
        img_var = tf.reshape(img_var, [batch_size] + list(dataset.image_shape))
        img_var = img_var[:rows * rows, :, :, :]
        imgs = tf.reshape(img_var, [rows, rows] + list(dataset.image_shape))
        stacked_img = []
        for row in xrange(rows):
            row_img = []
            for col in xrange(rows):
                row_img.append(imgs[row, col, :, :, :])
            stacked_img.append(tf.concat(axis=1, values=row_img))
        imgs = tf.concat(axis=0, values=stacked_img)
        imgs = tf.expand_dims(imgs, 0)
        tf.summary.image("image_%d_%s" % (dist_idx, dist.__class__.__name__), imgs)

log_vars =[]

input_tensor = tf.placeholder(tf.float32, [batch_size, dataset.image_dim])
with pt.defaults_scope(phase=pt.Phase.train):
    z_var = model.latent_dist.sample_prior(batch_size)
    with tf.variable_scope("generate") as scope:
        fake_x, _ = model.generate(z_var)
    #
    with tf.variable_scope("discriminate") as scope:
        real_d, _, _, _ = model.discriminate(input_tensor)
        scope.reuse_variables()
        fake_d, _, fake_reg_z_dist_info, _ = model.discriminate(fake_x)
    #   
    reg_z = model.reg_z(z_var)
    #
    discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
    generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))
    #
    log_vars.append(("discriminator_loss", discriminator_loss))
    log_vars.append(("generator_loss", generator_loss))
    #
    mi_est = tf.constant(0.)
    cross_ent = tf.constant(0.)
    #
    # compute for discrete and continuous codes separately
    # discrete:
    if len(model.reg_disc_latent_dist.dists) > 0:
        disc_reg_z = model.disc_reg_z(reg_z)
        disc_reg_dist_info = model.disc_reg_dist_info(fake_reg_z_dist_info)
        disc_log_q_c_given_x = model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
        disc_log_q_c = model.reg_disc_latent_dist.logli_prior(disc_reg_z)
        disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
        disc_ent = tf.reduce_mean(-disc_log_q_c)
        disc_mi_est = disc_ent - disc_cross_ent
        mi_est += disc_mi_est
        cross_ent += disc_cross_ent
        log_vars.append(("MI_disc", disc_mi_est))
        log_vars.append(("CrossEnt_disc", disc_cross_ent))
        discriminator_loss -= info_reg_coeff * disc_mi_est
        generator_loss -= info_reg_coeff * disc_mi_est
        #
    if len(model.reg_cont_latent_dist.dists) > 0:
        cont_reg_z = model.cont_reg_z(reg_z)
        cont_reg_dist_info = model.cont_reg_dist_info(fake_reg_z_dist_info)
        cont_log_q_c_given_x = model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
        cont_log_q_c = model.reg_cont_latent_dist.logli_prior(cont_reg_z)
        cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
        cont_ent = tf.reduce_mean(-cont_log_q_c)
        cont_mi_est = cont_ent - cont_cross_ent
        mi_est += cont_mi_est
        cross_ent += cont_cross_ent
        log_vars.append(("MI_cont", cont_mi_est))
        log_vars.append(("CrossEnt_cont", cont_cross_ent))
        discriminator_loss -= info_reg_coeff * cont_mi_est
        generator_loss -= info_reg_coeff * cont_mi_est
        #
    for idx, dist_info in enumerate(model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
        if "stddev" in dist_info:
            log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
            log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))
            #
    log_vars.append(("MI", mi_est))
    log_vars.append(("CrossEnt", cross_ent))
    #
    all_vars = tf.trainable_variables()
    d_vars = [var for var in all_vars if var.name.startswith('discriminate/d_')]
    g_vars = [var for var in all_vars if var.name.startswith('generate/g_')]
    #
    log_vars.append(("max_real_d", tf.reduce_max(real_d)))
    log_vars.append(("min_real_d", tf.reduce_min(real_d)))
    log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
    log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))
    #
    discriminator_optimizer = tf.train.AdamOptimizer(discriminator_learning_rate, beta1=0.5)
    discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                    var_list=d_vars)
    #
    generator_optimizer = tf.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
    generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)
    #
    for k, v in log_vars:
        tf.summary.scalar(k, v)

with pt.defaults_scope(phase=pt.Phase.test):
    with tf.variable_scope("generate", reuse=True) as scope:
        visualize_all_factors()


init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

counter = 0

log_keys = [x for x, _ in log_vars]
log_vars = [x for _, x in log_vars]

#--------------------# Generate Images #--------------------#

num_images = 100
root_gen_image_dir = '/gen/mnist/'
gen_image_dir = os.path.join(root_gen_image_dir, exp_name)


# Change the below values to generate different types of images:

cat_n = 3
total_n = 10
mean_nonreg_uniform = tf.random_normal([1,62])
stdv_nonreg_uniform = tf.random_normal([1,62])*0.125 + 0.125
mean_reg_u1 = tf.random_normal([1,1])
stdv_reg_u1 = tf.random_normal([1,1])*0.125 + 0.125
mean_reg_u2 = tf.random_normal([1,1])
stdv_reg_u2 = tf.random_normal([1,1])*0.125 + 0.125

# Converting to dist_info
prob_cat = np.zeros([1,total_n]) + TINY
prob_cat[0,cat_n] = 100
dist_flat_sample = model.latent_dist.activate_dist(tf.concat([mean_nonreg_uniform,\
	stdv_nonreg_uniform, prob_cat,mean_reg_u1, stdv_reg_u1, mean_reg_u2, stdv_reg_u2],1))

#sess.run(init)
saver.restore(sess, "/home/nithin127/infoGAN/ckt/mnist/mnist_2017_03_18_21_27_45/mnist_2017_03_18_21_27_45_5000.ckpt")

with tf.variable_scope("generate") as scope:
	#for i in range(num_images):
	scope.reuse_variables()
	img, _ = model.generate(model.latent_dist.sample(dist_flat_sample))
	img = sess.run(img)[0,:,:,0]

all_vars = tf.trainable_variables()
var_names = [var.name for var in all_vars]
var_names_g = [var.name for var in all_vars if var.name.startswith('g')]
