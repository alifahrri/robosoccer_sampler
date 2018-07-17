import numpy
import os, sys
import tensorflow as tf
from random import randint, random
# import pyximport; pyximport.install()
from util import *

class Dataset(object) :
    def __init__(self, *args, **kwargs):
        self.starts = []
        self.goals = []
        self.obstacles = []
        self.samples = []
        self.occ_grid = []
        self.c_var = None
        self.abspath = ''
        self.mode = 'load_all'
        self.str = []
        self.n_data = 0
        self.width = kwargs['width'] if 'width' in kwargs else WIDTH
        self.height = kwargs['height'] if 'height' in kwargs else HEIGHT
        self.cell = kwargs['cell'] if 'cell' in kwargs else CELL

    def parse(self, line) :
        str_arr = line.split()
        cvar_str = str_arr[0].translate(None, '()').replace(',', ' ').rstrip()
        samples_str = str_arr[1].translate(None, '()').replace(',', ' ').rstrip()
        cvars = [float(s) for s in cvar_str.split(' ')]
        samples = [float(s) for s in samples_str.split(' ')]
        start = numpy.array(cvars[:4])
        goal = numpy.array(cvars[4:8])
        obs = numpy.reshape(cvars[8:],[9,2])
        grid = grid_map(obs, self.cell, self.width, self.height, self.cell)
        samples = numpy.reshape(samples, [len(samples)/4, 4])
        ret = [[obs, start, goal, samples[int(i)], numpy.concatenate(grid)] for i in numpy.linspace(0, len(samples)-1, 3)]
        return ret
    
    def get(self, i) :
        if self.mode == 'load_partial' :
            # value = self.parse(self.str[i])[0]
            value = parse(self.str[i], self.cell, self.width, self.height)
            return value[0][1], value[0][2], value[0][0], [v[3] for v in value]
        elif self.mode == 'load_all' :
            return self.starts[i], self.goals[i], self.obstacles[i], self.samples[i]

    def get_data(self, train_test_ratio) :
        idx = int(len(self.samples)*train_test_ratio)
        data = {
            'train' : {
                'conditions' : self.c_var[:idx],
                'samples' : self.samples[:idx]
            },
            'test' : {
                'conditions' : self.c_var[-idx:],
                'samples' : self.samples[-idx:]
            }
        }
        return data

    def get_data_from_str(self, i) :
        return parse_sample_condition(self.str[i], self.cell, self.width, self.height)
    
    def load(self, filename) :
        statinfo = os.stat(filename)
        if statinfo.st_size > 4000000 :
            self.mode = 'load_partial'
        self.abspath = os.path.abspath(filename)
        with open(filename) as f :
            for line in f.readlines() :
                if self.mode == 'load_all' :
                    value = self.parse(line)
                    for v in value :
                        self.obstacles.append(v[0])
                        self.starts.append(v[1])
                        self.goals.append(v[2])
                        self.samples.append(v[3])
                        self.occ_grid.append(v[4])
                elif self.mode == 'load_partial' :
                    self.str.append(line)
        if self.mode == 'load_all' :
            self.c_var = numpy.concatenate((self.starts[:], self.goals[:], self.occ_grid[:]), axis=1)
            # print 'samples len :', len(self.samples)
            # print 'sample shape :', self.samples[0].shape
            # print 'c_var shape :', self.c_var.shape
            # print 'c_var[0]:', self.c_var[0]
            # print 'c_var[1]:', self.c_var[1]
        self.n_data = len(self.samples) if self.mode == 'load_all' else len(self.str)
        return self.n_data

class CVAESampler(object) :
    def __init__(self, *args, **kwargs) :
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config)

        self.train_iter = kwargs['epoch'] if 'epoch' in kwargs else 500000
        self.width = kwargs['width'] if 'width' in kwargs else WIDTH
        self.height = kwargs['height'] if 'height' in kwargs else HEIGHT
        self.cell = kwargs['cell'] if 'cell' in kwargs else CELL
        self.model_dir = kwargs['model_dir'] if 'model_dir' in kwargs else './models'

        self.network = self.create_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.it = 0
        self.mb_size = 512
        if 'mb_size' in kwargs : self.mb_size = kwargs['mb_size']

        if os.path.isfile(self.model_dir+'.index') :
            self.restore()
    
    def signal_handler(self, signal, frame):
        print 'signal', signal, 'received'
        self.save()
        sys.exit(0)

    def restore(self) :
        print 'restoring', self.model_dir
        self.saver.restore(self.sess, self.model_dir)
        with open('it.txt', 'r') as f :
            s = f.read()
            self.it = int(s)

    def save(self) :
        print 'training done! now saving in %s' %self.model_dir
        self.saver.save(self.sess,self.model_dir, write_meta_graph=True)
        with open('it.txt', 'w+') as f :
            f.write('%s' %self.it)

    def sample(self, n_sample, xs, xg, obs) :
        # grid = numpy.concatenate(grid_map(obs, self.cell, self.width, self.height, self.cell))
        # c_var = numpy.concatenate((xs, xg, grid))
        z = numpy.random.random(n_sample*self.network['z_dim'])
        # y, z = self.sess.run([self.network['y'], self.network['z']], feed_dict={
        #     # self.network['z'] : numpy.random.randn(n_sample, self.network['z_dim']),
        #     self.network['z'] : numpy.reshape(z,(n_sample, self.network['z_dim'])),
        #     self.network['c'] : numpy.repeat([c_var], n_sample, axis=0)
        # })
        grid = grid_map(obs, self.cell, self.width, self.height, self.cell)
        states = numpy.concatenate((xs, xg))
        y, z = self.sess.run([self.network['y'], self.network['z']], feed_dict={
            # self.network['z'] : numpy.random.randn(n_sample, self.network['z_dim']),
            self.network['z'] : numpy.reshape(z,(n_sample, self.network['z_dim'])),
            self.network['c_grid'] : numpy.repeat([grid], n_sample, axis=0),
            self.network['states'] : numpy.repeat([states], n_sample, axis=0)
        })
        return y, z

    # def train_step(self, x_mb, c_mb, c_grid_mb) :
    def train_step(self, x_mb, states_mb, c_grid_mb) :
        s = [len(c_grid_mb), len(c_grid_mb[0]), len(c_grid_mb[0][0])]
        _, loss = self.sess.run([self.network['train_op'], self.network['cvae_loss']], feed_dict={
            self.network['X']: x_mb, 
            # self.network['c']: c_mb,
            self.network['states'] : states_mb,
            self.network['c_grid'] : numpy.reshape(c_grid_mb, [s[0], s[1], s[2], 1])
        })

        if self.it % 100 == 0:
            print('Iter: {}'.format(self.it))
            print('Loss: {:.4}'. format(loss))
            print()

    def shuffled_data(self, mb_size, x_train, c_train) :
        batch_elements = [randint(0,len(x_train)-1) for n in range(0,mb_size)]
        x_mb = [x_train[i] for i in batch_elements]
        c_mb = [c_train[i] for i in batch_elements]
        return x_mb, c_mb

    def shuffled_data_from_str(self, mb_size, strings, n_data) :
        batch_elements = [randint(0,n_data-1) for n in range(0,mb_size)]
        # x_mb, c_mb = parse_samples_conditions(strings, batch_elements, self.cell, self.width, self.height)
        # x_mb, c_mb = parse_samples_conditions(strings, batch_elements, self.cell, self.width, self.height)
        x_mb, c_mb, g_mb = parse_lines(strings, batch_elements, self.cell, self.width, self.height)
        return x_mb, c_mb, g_mb

    def train_from_str(self, strings, n_data) :
        mb_size = self.mb_size
        it = self.it
        train_iter = self.train_iter
        for it in range(it,train_iter) :
            # randomly generate batches
            # x_mb, c_mb = self.shuffled_data_from_str(mb_size, strings, n_data)
            x_mb, c_mb, g_mb = self.shuffled_data_from_str(mb_size, strings, n_data)
            self.it = it
            self.train_step(x_mb, c_mb, g_mb)

    def train(self, x_train, c_train) :
        mb_size = self.mb_size
        it = self.it
        train_iter = self.train_iter
        for it in range(it,it+train_iter) :
            # randomly generate batches
            # x_mb, c_mb = self.shuffled_data(mb_size, x_train, c_train)
            x_mb, c_mb, g_mb = self.shuffled_data_from_str(mb_size, strings, n_data)
            self.it = it
            self.train_step(x_mb, c_mb, g_mb)

    def create_network(self) :
        # neural network parameters
        h_Q_dim = 512
        h_P_dim = 512

        c = 0
        lr = 1e-4

        # problem dimensions
        n_obs = 9
        dim = 4

        z_dim = 2 # latent
        X_dim = dim # samples
        y_dim = dim # reconstruction of the original point
        c_dim = self.width * self.height / (self.cell * self.cell) + 2 * dim
        c_grid_dim = [None, self.width / self.cell, self.height / self.cell, 1]

        # define networks
        X = tf.placeholder(tf.float32, shape=[None, X_dim])
        # c = tf.placeholder(tf.float32, shape=[None, c_dim])

        states_dim = 2 * dim
        c_grid = tf.placeholder(tf.float32, shape=c_grid_dim)
        states = tf.placeholder(tf.float32, shape=[None, states_dim])
        conv1 = tf.layers.conv2d(
                inputs=c_grid,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="conv1")
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="conv2")
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")
        pool2_flat = tf.reshape(pool2, [-1, 11 * 7 * 64], name="pool2_flat")
        grid_dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="grid_dense")

        c = tf.concat(axis=1, values=[grid_dense, states])

        # Q
        inputs_Q = tf.concat(axis=1, values=[X,c])

        dense_Q1 = tf.layers.dense(inputs=inputs_Q, units=h_Q_dim, activation=tf.nn.relu)
        dropout_Q1 = tf.layers.dropout(inputs=dense_Q1, rate=0.5)
        dense_Q2 = tf.layers.dense(inputs=dropout_Q1, units=h_Q_dim, activation=tf.nn.relu)

        z_mu = tf.layers.dense(inputs=dense_Q2, units=z_dim) # output here is z_mu
        z_logvar = tf.layers.dense(inputs=dense_Q2, units=z_dim) # output here is z_logvar

        # P
        eps = tf.random_normal(shape=tf.shape(z_mu))
        z = z_mu + tf.exp(z_logvar / 2) * eps
        inputs_P = tf.concat(axis=1, values=[z,c])

        dense_P1 = tf.layers.dense(inputs=inputs_P, units=h_P_dim, activation=tf.nn.relu)
        dropout_P1 = tf.layers.dropout(inputs=dense_P1, rate=0.5)
        dense_P2 = tf.layers.dense(inputs=dropout_P1, units=h_P_dim, activation=tf.nn.relu)

        y = tf.layers.dense(inputs=dense_P2, units=X_dim) # fix to also output y

        # training
        w = [[1, 1, 0.5, 0.5]];
        recon_loss = tf.losses.mean_squared_error(labels=X, predictions=y, weights=w)
        # TODO: fix loss function for angles going around
        kl_loss = 10**-4 * 2 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        cvae_loss = tf.reduce_mean(kl_loss + recon_loss)
        train_op = tf.train.AdamOptimizer(lr).minimize(cvae_loss)

        net = {
            'z_dim' : z_dim,
            'X' : X,
            'c' : c,
            'conv1' : conv1, # edit
            'pool1' : pool1, # edit
            'conv2' : conv2, # edit
            'pool2' : pool2, # edit
            'pool2_flat' : pool2_flat, # edit
            'grid_dense' : grid_dense, # edit
            'c_grid' : c_grid, # edit
            'states' : states, # edit
            'inputs_Q' : inputs_Q,
            'inputs_P' : inputs_P,
            'dense_Q1' : dense_Q1,
            'dropout_Q1' : dropout_Q1,
            'dense_Q2' : dense_Q2,
            'z_mu' : z_mu,
            'z_logvar' : z_logvar,
            'eps' : eps,
            'z' : z,
            'inputs_P' : inputs_P,
            'dense_P1' : dense_P1,
            'dropout_P1' : dropout_P1,
            'dense_P2' : dense_P2,
            'y' : y,
            'recon_loss' : recon_loss,
            'kl_loss' : kl_loss,
            'cvae_loss' : cvae_loss,
            'train_op' : train_op
        }

        return net
