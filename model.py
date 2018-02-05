# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import flags as fl
import time

import sys, os
scriptPath = os.path.abspath(sys.argv[0])
sys.path.append(os.path.dirname(scriptPath))

class Model(object):
    def __init__(self, batch_size, num_epochs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.uid = tf.placeholder(tf.float32, [None, fl.uid_dimension], name="uid")
        self.sex = tf.placeholder(tf.float32, [None, fl.sex_dimension], name="sex")
        self.age = tf.placeholder(tf.float32, [None, fl.age_dimension], name="age")
        self.occ = tf.placeholder(tf.float32, [None, fl.occ_dimension], name="occ")

        # self.uid_layer = tf.Variable(tf.random_uniform(
        #     [fl.uid_dimension, fl.embed_dimension], -1, 1), name="uid_layer")
        # self.sex_layer = tf.Variable(tf.random_uniform(
        #     [fl.sex_dimension, fl.embed_dimension], -1, 1), name="sex_layer")
        # self.age_layer = tf.Variable(tf.random_uniform(
        #     [fl.age_dimension, fl.embed_dimension], -1, 1), name="age_layer")
        # self.occ_layer = tf.Variable(tf.random_uniform(
        #     [fl.occ_dimension, fl.embed_dimension], -1, 1), name="occ_layer")

        self.mid = tf.placeholder(tf.float32, [None, fl.mid_dimension], name="mid")
        self.category = tf.placeholder(tf.float32, [None, fl.category_dimension], name="category")
        self.movie_name = tf.placeholder(tf.int32, [None, 15], name="movie_name")

        # self.mid_layer = tf.Variable(tf.random_uniform([fl.mid_dimension, fl.embed_dimension], -1, 1), name="mid_layer")
        # self.catetory_layer = tf.Variable(tf.random_uniform([fl.category_dimension, fl.embed_dimension], -1, 1), name="category_layer")

        with tf.name_scope("movie_embedding"):
            self.movie_name_matrix = tf.Variable(tf.random_uniform(
                [fl.title_dimension, fl.embed_dimension], -1, 1), name="movie_name_matrix")
            self.movie_name_layer = tf.nn.embedding_lookup(self.movie_name_matrix, self.movie_name, name="movie_name_layer")
            self.movie_name_layer = tf.expand_dims(self.movie_name_layer, -1)

        self.targets = tf.placeholder(tf.float32, [None, 1], name="targets")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def show_shape(self, tensor):
        print ('%s.get_shape():' % tensor.name, tensor.get_shape())

    def userRefrence(self):
        # 全连接层
        user_combine_layer = tf.concat([self.uid, self.sex, self.age, self.occ], 1)
        user_combine_layer = tf.cast(user_combine_layer, tf.float32)
        self.show_shape(user_combine_layer)
        user_combine_layer = tf.layers.dense(user_combine_layer, 200, tf.sigmoid)
        user_combine_layer = tf.reshape(user_combine_layer, [-1, 200])
        return user_combine_layer

    def movieRefrence(self):
        # 卷积层
        pool_layer_list = []
        window_sizes = (2, 3, 4, 5)
        # TODO: 尝试一种卷积核
        for size in window_sizes:
            name = 'title_conv_max_pool_%d' % size
            with tf.name_scope(name):
                filter_weights = tf.Variable(tf.truncated_normal([size, fl.embed_dimension, 1, fl.filter_num], stddev=0.1), name="filter_weights")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[fl.filter_num]), name="filter_bias")

                conv_layer = tf.nn.conv2d(self.movie_name_layer, filter_weights, [1, 1, 1, 1], padding="VALID", name="conv_layer")
                relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")
                maxpool_layer = tf.nn.max_pool(relu_layer, [1, fl.sentences_size - size + 1, 1, 1], [1, 1, 1, 1], padding="VALID", name="maxpool_layer")
                pool_layer_list.append(maxpool_layer)

        with tf.name_scope("pool_dropout"):
            pool_layer = tf.concat(pool_layer_list, 3, name="pool_layer")
            max_num = len(window_sizes) * fl.filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, max_num], name="pool_layer_flat")
            self.movie_drop_layer = tf.nn.dropout(pool_layer_flat, self.dropout_keep_prob, name="dropout_layer")
        
        with tf.name_scope("movie_fc"):
            movie_combine_layer = tf.concat([self.mid, self.category, self.movie_drop_layer], 1)
            self.show_shape(movie_combine_layer)
            movie_combine_layer = tf.cast(movie_combine_layer, tf.float32)
            movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.sigmoid)
            movie_combine_layer = tf.reshape(movie_combine_layer, [-1, 200], name="movie_combine_layer_reshape")
        return movie_combine_layer

    def inference(self):
        user_combine_layer = self.userRefrence()
        movie_combine_layer = self.movieRefrence()
        with tf.name_scope("inference"):
            inference = tf.matmul(user_combine_layer, tf.transpose(movie_combine_layer))
        with tf.name_scope("loss"):
            cost = tf.losses.mean_squared_error(self.targets, inference)
            loss = tf.reduce_mean(cost)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)
        return train_op, loss, gradients, global_step

def test():
    model = Model()
    model.inference()

if __name__ == '__main__':
    test()
