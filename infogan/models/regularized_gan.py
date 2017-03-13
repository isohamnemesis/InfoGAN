from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
from infogan.misc.custom_ops import leaky_rectify


class RegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, batch_size, image_shape, network_type):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

    def create_discriminator(self, inp):
        image_size = self.image_shape[0]
        if self.network_type == "mnist":
            with tf.variable_scope("d_net"):
                with tf.variable_scope("conv1") as scope:
                    inp_ = tf.reshape(inp,[-1]+list(self.image_shape))
                    kernel = tf.get_variable('weight',[4,4,self.image_shape[-1],64], \
                        initializer = tf.truncated_normal_initializer(5e-2))
                    conv = tf.nn.conv2d(inp_,kernel,[1,1,1,1], padding='SAME')
                    bias = tf.get_variable('bias',[64], initializer=tf.constant_initializer(0.1))
                    pre_act = tf.nn.bias_add(conv,bias)
                    # applying batch_norm
                    pre_act = tf.nn.batch_normalization(pre_act)
                    # leaky ReLu with alpha = 0.01
                    conv1 = tf.maximum(0.01*pre_act, pre_act, name=scope.name)

                pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

                with tf.variable_scope("conv2") as scope:
                    kernel = tf.get_variable('weight',[4,4,64,128], \
                        initializer = tf.truncated_normal_initializer(5e-2))
                    conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1], padding='SAME')
                    bias = tf.get_variable('bias',[128], initializer=tf.constant_initializer(0.1))
                    pre_act = tf.nn.bias_add(conv,bias)
                    # applying batch_norm
                    pre_act = tf.nn.batch_normalization(pre_act)
                    # leaky ReLu with alpha = 0.01
                    conv2 = tf.maximum(0.01*pre_act, pre_act, name=scope.name)

                pool2 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')

                with tf.variable_scope("fc3") as scope:
                    inp_ = tf.reshape(pool2, [self.batch_size, -1])
                    dim = inp.get_shape()[1].value
                    weights = tf.get_variable('weight',[dim,1024],\
                        initializer=tf.truncated_normal_initializer(4e-3))
                    bias = tf.get_variable('bias',[1024],initializer=tf.constant_initializer(0.1))
                    pre_act = tf.matmul(inp_,weights) + bias
                    # applying batch_norm
                    pre_act = tf.nn.batch_normalization(pre_act)
                    # leaky ReLu with alpha = 0.01
                    fc3 = tf.maximum(0.01*pre_act, pre_act, name=scope.name)

                with tf.variable_scope("fc4") as scope:
                    weights = tf.get_variable('weight',[1024,128],\
                        initializer=tf.truncated_normal_initializer(4e-3))
                    bias = tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.1))
                    pre_act = tf.nn.bias_add(tf.matmul(fc3,weights),bias,name=scope.name)
                    # applying batch_norm
                    pre_act = tf.nn.batch_normalization(pre_act)
                    # leaky ReLu with alpha = 0.01
                    fc4 = tf.maximum(0.01*pre_act, pre_act, name=scope.name)

                with tf.variable_scope("d_temp") as scope:
                    weights = tf.get_variable('weight',[1024,1],\
                        initializer=tf.truncated_normal_initializer(4e-3))
                    bias = tf.get_variable('bias',[1],initializer=tf.constant_initializer(0.1))
                    d_temp = tf.nn.bias_add(tf.matmul(fc3,weights),bias,name=scope.name)
                    
                self.discriminator_template = d_temp

                with tf.variable_scope("enc_temp") as scope:
                    weights = tf.get_variable('weight',[128,self.reg_latent_dist.dist_flat_dim],\
                        initializer=tf.truncated_normal_initializer(4e-3))
                    bias = tf.get_variable('bias',[self.reg_latent_dist.dist_flat_dim],\
                        initializer=tf.constant_initializer(0.1))
                    enc_temp = tf.nn.bias_add(tf.matmul(fc4,weights),bias,name=scope.name)

                self.encoder_template = enc_temp
        else:
            raise NotImplementedError

    def create_generator(self, inp):
        image_size = self.image_shape[0]
        if self.network_type == "mnist":
            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(image_size / 4 * image_size / 4 * 128).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size / 4, image_size / 4, 128]).
                     custom_deconv2d([0, image_size / 2, image_size / 2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
                     flatten())
        else:
            raise NotImplementedError


    def discriminate(self, x_var):
        self.create_discriminator(input=x_var)
        d_out = self.discriminator_template.construct(input=x_var)
        d = tf.nn.sigmoid(d_out[:, 0])
        reg_dist_flat = self.encoder_template
        reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
        return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate(self, z_var):
        self.create_generator(z_var)
        x_dist_flat = self.generator_template
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info

    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)
