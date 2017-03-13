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

    def custom_batch_norm(self, input_layer, epsilon=1e-5):
        shape = input_layer.shape
        shp = shape[-1]
        mean, variance = tf.nn.moments(input_layer, [0])
        return tf.nn.batch_normalization( input_layer, mean, variance, None, None, epsilon)

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
                    pre_act = self.custom_batch_norm(pre_act)
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
                    pre_act = self.custom_batch_norm(pre_act)
                    # leaky ReLu with alpha = 0.01
                    conv2 = tf.maximum(0.01*pre_act, pre_act, name=scope.name)

                pool2 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')

                with tf.variable_scope("fc3") as scope:
                    inp_ = tf.reshape(pool2, [self.batch_size, -1])
                    dim = inp_.get_shape()[1].value
                    weights = tf.get_variable('weight',[dim,1024],\
                        initializer=tf.truncated_normal_initializer(4e-3))
                    bias = tf.get_variable('bias',[1024],initializer=tf.constant_initializer(0.1))
                    pre_act = tf.matmul(inp_,weights) + bias
                    # applying batch_norm
                    pre_act = self.custom_batch_norm(pre_act)
                    # leaky ReLu with alpha = 0.01
                    fc3 = tf.maximum(0.01*pre_act, pre_act, name=scope.name)

                with tf.variable_scope("fc4") as scope:
                    weights = tf.get_variable('weight',[1024,128],\
                        initializer=tf.truncated_normal_initializer(4e-3))
                    bias = tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.1))
                    pre_act = tf.nn.bias_add(tf.matmul(fc3,weights),bias,name=scope.name)
                    # applying batch_norm
                    pre_act = self.custom_batch_norm(pre_act)
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
                with tf.variable_scope("fc1") as scope:
                    output_shape = [inp.shape[-1].value,1024]
                    weights = tf.get_variable('weight', output_shape,\
                        initializer=tf.truncated_normal_initializer(0.1))
                    bias = tf.get_variable('bias',[1024],initializer=tf.constant_initializer(0.1))
                    pre_act = tf.nn.bias_add(tf.matmul(inp,weights),bias)
                    # applying batch_norm
                    pre_act = self.custom_batch_norm(pre_act)
                    # applying activation
                    fc1 = tf.nn.relu(pre_act, name=scope.name)

                with tf.variable_scope("fc2") as scope:
                    output_shape = [-1, image_size/4*image_size/4*128]
                    weights = tf.get_variable('weight',[1024,output_shape[-1]],initializer=tf.truncated_normal_initializer(0.1))
                    bias = tf.get_variable('bias',[output_shape[-1]],initializer=tf.constant_initializer(0.1))
                    pre_act = tf.nn.bias_add(tf.matmul(fc1,weights),bias)
                    # applying batch_norm
                    pre_act = self.custom_batch_norm(pre_act)
                    # applying activation
                    fc2 = tf.nn.relu(pre_act, name=scope.name)
                    fc2_r = tf.reshape(fc2,[-1, image_size/4, image_size/4, 128])

                with tf.variable_scope("deconv3") as scope:
                    output_shape = [-1, image_size/2, image_size/2, 64]
                    kernel = tf.get_variable('weight', [4, 4, 64, 128],\
                              initializer=tf.random_normal_initializer(stddev=0.02))
                    deconv = tf.nn.conv2d_transpose(fc2_r, kernel,\
                        output_shape=output_shape,strides=[1, 2, 2, 1])

                    bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                    deconv = tf.reshape(tf.nn.bias_add(deconv, bias), output_shape)
                    # applying batch_norm
                    pre_act = self.custom_batch_norm(deconv)
                    # applying activation
                    deconv3 = tf.nn.relu(pre_act, name=scope.name)

                with tf.variable_scope("deconv4") as scope:
                    output_shape = [-1, image_size, image_size, 1]
                    kernel = tf.get_variable('weight', [4, 4, 1, 64],\
                              initializer=tf.random_normal_initializer(stddev=0.02))
                    deconv = tf.nn.conv2d_transpose(deconv3, kernel,\
                        output_shape=output_shape,strides=[1, 2, 2, 1])

                    bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                    deconv = tf.reshape(tf.nn.bias_add(deconv, bias), output_shape)
                    # applying batch_norm
                    pre_act = self.custom_batch_norm(deconv)
                    # applying activation
                    deconv4 = tf.nn.relu(pre_act, name=scope.name)

                self.generator_template = deconv4
        else:
            raise NotImplementedError


    def discriminate(self, x_var):
        self.create_discriminator(inp=x_var)
        d_out = self.discriminator_template
        d = tf.nn.sigmoid(d_out[:, 0])
        reg_dist_flat = self.encoder_template
        reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
        return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate(self, z_var):
        self.create_generator(inp=z_var)
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

#Note: Change all the tf.batch_normalization statements to the ones in custom_ops
