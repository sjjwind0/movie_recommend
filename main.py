# -*- encoding:utf-8 -*-

import os, shutil

import tensorflow as tf
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import re
import datetime
import ast

UID_MAX = 6042
SEX_MAX = 2
AGE_MAX = 7
OCC_MAX = 21
ADDR_MAX = 3500
MID_MAX = 3980
NAME_MAX = 6000
CATEGORY_MAX = 19

NAME_WORD_MAX = 15

EMBEDDING_DIMENSION = 64
NAME_CONV_KERNEL_COUNT = 16

USER_OUTPUT_DIMENSION = 300

EPOCHS_COUNT = 20
BATCH_SIZE = 256
DROPOUT_KEEP_PROP = 0.95
LEARNING_RATE = 1e-3
SAVE_DIR = "./train_data"
REPLATIONSHIP_DIR = "./relation"

class Model(object):
    def __init__(self, enable_debug = True):
        self.enable_debug = enable_debug

    def showTensors(self, name, tensor):
        if self.enable_debug:
            if type(tensor) == list or type(tensor) == tuple:
                for t in tensor:
                    print ("%s.%s.get_shape():" % (name, t.name), t.get_shape())
            else:
                print ("%s.%s.get_shape():" % (name, tensor.name), tensor.get_shape())

    def initTensors(self):
        # user layers
        self.uid = tf.placeholder(tf.int32, [None, 1], name = "uid")
        self.sex = tf.placeholder(tf.int32, [None, 1], name = "sex")
        self.age = tf.placeholder(tf.int32, [None, 1], name = "age")
        self.occ = tf.placeholder(tf.int32, [None, 1], name = "occ")
        self.addr = tf.placeholder(tf.int32, [None, 1], name = "addr")

        # movie layer
        self.mid = tf.placeholder(tf.int32, [None, 1], name = "mid")
        # 电影最多有15个单词
        self.name = tf.placeholder(tf.int32, [None, 15], name = "name")
        # 电影类型最多有18个类型
        self.category = tf.placeholder(tf.int32, [None, 18], name = "category")

        # target
        self.target = tf.placeholder(tf.int32, [None, 1], name = "target")

        # learning rate
        self.learning_rate = tf.placeholder(tf.float32, name = "learning_rate")

        # keep dropout rate
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prop")

    def build_embedding_layer(self, input_layer, shape, name):
        with tf.name_scope(name + "_embedding"):
            matrix = tf.Variable(tf.random_uniform(shape, -1, 1), name = name + "embedding_matrix")
            embedding_layer = tf.nn.embedding_lookup(matrix, input_layer, name = name + "embedding_layer")
        return embedding_layer

    def buildUserLayers(self):
        with tf.name_scope("user_embedding_layer"):
            uid_embedding_layer = self.build_embedding_layer(self.uid, [UID_MAX, EMBEDDING_DIMENSION], name = "uid")
            sex_embedding_layer = self.build_embedding_layer(self.sex, [AGE_MAX, EMBEDDING_DIMENSION // 2], name = "sex")
            age_embedding_layer = self.build_embedding_layer(self.age, [AGE_MAX, EMBEDDING_DIMENSION // 2], name = "age")
            occ_embedding_layer = self.build_embedding_layer(self.occ, [OCC_MAX, EMBEDDING_DIMENSION // 2], name = "occ")
            addr_embedding_layer = self.build_embedding_layer(self.addr, [ADDR_MAX, EMBEDDING_DIMENSION], name = "addr")
            self.showTensors("user_layer", [uid_embedding_layer, sex_embedding_layer, age_embedding_layer, occ_embedding_layer, addr_embedding_layer])

        with tf.name_scope("user_fc"):
            # 各个embedding层的dense层
            uid_fc_layer = tf.layers.dense(uid_embedding_layer, EMBEDDING_DIMENSION, name = "uid_fc_layer", activation=tf.nn.relu)
            sex_fc_layer = tf.layers.dense(sex_embedding_layer, EMBEDDING_DIMENSION, name = "sex_fc_layer", activation=tf.nn.relu)
            age_fc_layer = tf.layers.dense(age_embedding_layer, EMBEDDING_DIMENSION, name = "age_fc_layer", activation=tf.nn.relu)
            occ_fc_layer = tf.layers.dense(occ_embedding_layer, EMBEDDING_DIMENSION, name = "occ_fc_layer", activation=tf.nn.relu)
            addr_fc_layer = tf.layers.dense(addr_embedding_layer, EMBEDDING_DIMENSION, name = "addr", activation=tf.nn.relu)
            self.showTensors("user_fc", [uid_fc_layer, sex_fc_layer, age_fc_layer, occ_fc_layer, addr_fc_layer])

            # 把各个dense层全连接起来

            # [None, 1, EMBEDDING_DIMENSION * 5]
            user_fc_layer = tf.concat([uid_fc_layer, sex_fc_layer, age_fc_layer, occ_fc_layer, addr_fc_layer], axis=2, name = "user_fc_layer_concat")
            user_fc_layer = tf.contrib.layers.fully_connected(user_fc_layer, USER_OUTPUT_DIMENSION, tf.nn.sigmoid)

            # reshape to [None, USER_OUTPUT_DIMENSION]
            user_fc_layer = tf.reshape(user_fc_layer, [-1, 1, USER_OUTPUT_DIMENSION], name = "user_fc_layer_reshape")

            self.showTensors("user_fc", user_fc_layer)

        return user_fc_layer

    def buildMovieLayers(self):
        with tf.name_scope("movie_name_embedding"):
            matrix = tf.Variable(tf.random_uniform([NAME_MAX, EMBEDDING_DIMENSION], -1, 1), name = "movie_name_embedding_matrix")
            movie_name_embedding_layer = tf.nn.embedding_lookup(matrix, self.name, name = "movie_name_embedding_layer")
            movie_name_embedding_layer = tf.expand_dims(movie_name_embedding_layer, -1)
            self.showTensors("movie_name_embedding", movie_name_embedding_layer)

        # 电影名的多卷积层
        conv_layer_list = []
        with tf.name_scope("movie_name_conv"):
            for kernel_size in range(2, 6):
                with tf.name_scope("movie_name_conv_%d" % kernel_size):
                    # [kernel_size, 32, 1, 8]
                    W = tf.Variable(tf.truncated_normal([kernel_size, EMBEDDING_DIMENSION, 1, NAME_CONV_KERNEL_COUNT], stddev=0.1), name = "movie_name_W")
                    B = tf.Variable(tf.constant(0.1, shape = [NAME_CONV_KERNEL_COUNT]), name = "movie_name_B")

                    # conv
                    conv_layer = tf.nn.conv2d(movie_name_embedding_layer, W, [1, 1, 1, 1], padding = "VALID", name = "movie_name_conv_layer")
                    conv_layer = tf.nn.bias_add(conv_layer, B, name = "movie_name_conv_layer_bias")
                    relu_layer = tf.nn.relu(conv_layer, name = "movie_name_relu_layer")
                    self.showTensors("movie_name_conv_%d" %  kernel_size, relu_layer)

                    # pool
                    pool_layer = tf.nn.max_pool(relu_layer, [1, NAME_WORD_MAX - kernel_size + 1, 1, 1], [1, 1, 1, 1], padding = "VALID", name = "movie_name_pool_layer")
                    conv_layer_list.append(pool_layer)
        self.showTensors("movie_name_conv", conv_layer_list)
        with tf.name_scope("movie_name_output"):
            conv_layer = tf.concat(conv_layer_list, 3, name = "movie_name_all_conv_layer")
            self.showTensors("movie_name_output", conv_layer)
            # 一共有5个卷积层，每个卷积层有NAME_CONV_KERNEL_COUNT个卷积核，输出共5 * NAME_CONV_KERNEL_COUNT
            conv_layer = tf.reshape(conv_layer, [-1, 1, 4 * NAME_CONV_KERNEL_COUNT])

        with tf.name_scope("movie_name_dropout_layer"):
            conv_layer = tf.nn.dropout(conv_layer, DROPOUT_KEEP_PROP)

        with tf.name_scope("movie_embedding_layer"):
            mid_embedding_layer = self.build_embedding_layer(self.mid, [MID_MAX, EMBEDDING_DIMENSION], name = "mid")
            category_mebdding_layer = self.build_embedding_layer(self.category, [CATEGORY_MAX, EMBEDDING_DIMENSION], name="category")
            category_mebdding_layer = tf.reduce_sum(category_mebdding_layer, axis = 1, keepdims = True)

        with tf.name_scope("movie_fc"):
            mid_fc_layer = tf.layers.dense(mid_embedding_layer, EMBEDDING_DIMENSION, name = "mid_fc_layer", activation = tf.nn.relu)
            category_fc_layer = tf.layers.dense(category_mebdding_layer, EMBEDDING_DIMENSION, name = "category_fc_layer", activation = tf.nn.relu)

            movie_layer = tf.concat([mid_fc_layer, category_fc_layer, conv_layer], 2)
            movie_layer = tf.contrib.layers.fully_connected(movie_layer, USER_OUTPUT_DIMENSION, tf.sigmoid)

            movie_layer = tf.reshape(movie_layer, [-1, 1, USER_OUTPUT_DIMENSION], name = "movie_fc_layer_reshape")
        return movie_layer

    def inference(self, graph):
        self.initTensors()
        user_layer = self.buildUserLayers()
        movie_layer = self.buildMovieLayers()
        with tf.name_scope("inference"):
            inference = tf.matmul(user_layer, tf.matrix_transpose(movie_layer))
            self.showTensors("inference_1", inference)
            inference = tf.reshape(inference, [-1, 1], name="inference")
            self.showTensors("inference_2", inference)
        with tf.name_scope("loss"):
            cost = tf.losses.mean_squared_error(self.target, inference)
            loss = tf.reduce_sum(cost, name = "loss")

        global_step = tf.Variable(0, name = "global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step = global_step)

        return train_op, inference, loss, gradients, global_step

    def getTensors(self, graph):
        uid = graph.get_tensor_by_name("uid:0")
        sex = graph.get_tensor_by_name("sex:0")
        age = graph.get_tensor_by_name("age:0")
        occ = graph.get_tensor_by_name("occ:0")
        addr = graph.get_tensor_by_name("addr:0")

        mid = graph.get_tensor_by_name("mid:0")
        category = graph.get_tensor_by_name("category:0")
        name = graph.get_tensor_by_name("name:0")
        target = graph.get_tensor_by_name("target:0")

        inference = graph.get_tensor_by_name("inference/inference:0")
        user_layer = graph.get_tensor_by_name("user_fc/user_fc_layer_reshape:0")
        movie_layer = graph.get_tensor_by_name("movie_fc/movie_fc_layer_reshape:0")

        learning_rate = graph.get_tensor_by_name("learning_rate:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prop:0")
        return uid, sex, age, occ, addr, mid, category, name, target, inference, user_layer, movie_layer, learning_rate, dropout_keep_prob

class DataInpter(object):
    def __init__(self, path):
        self.work_dir = os.path.abspath(path)
        self.age_map = None
        self.addr_map = None
        self.category_map = None
        self.mid_map = None
        self.name_map = None
        self.name_year_map = None
        self.users_values = []
        self.movies_values = []
        self.raw_users_values = None
        self.raw_movies_values = None
        self.resoreAllReplationship()

    def readData(self):
        need_save_relationship = not self.resoreAllReplationship()
        users_title = ["mid", "sex", "age", "occ", "addr"]
        users = pd.read_table(self.work_dir + "/users.dat", sep = "::", header = None, names = users_title, engine = "python")
        users = users.filter(regex = "mid|sex|age|occ|addr")
        self.raw_users_values = users.values

        users["sex"] = users["sex"].map({"F": 0, "M": 1})

        if need_save_relationship:
            self.age_map = {val: ii for ii, val in enumerate(set(users["age"]))}
        users["age"] = users["age"].map(self.age_map)

        #地区码转数字字段
        if need_save_relationship:
            addr2int = {val: ii for ii, val in enumerate(set(users["addr"]))}
            self.addr_map = {val: addr2int[val] for ii, val in enumerate(set(users["addr"]))}
        users["addr"] = users["addr"].map(self.addr_map)

        self.users_values = users.values

        movies_title = ["mid", "name", "category"]
        movies = pd.read_table(self.work_dir + "/movies.dat", sep = "::", header = None, names = movies_title, engine = "python")
        self.raw_movies_values = movies.values

        if need_save_relationship:
            self.mid_map = {val[0]: i for i, val in enumerate(movies.values)}
        #将name中的年份去掉
        if need_save_relationship:
            pattern = re.compile(r"^(.*)\((\d+)\)$")
            self.name_year_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies["name"]))}
        movies["name"] = movies["name"].map(self.name_year_map)

        #电影类型转数字字典
        if need_save_relationship:
            category_set = set()
            for val in movies["category"].str.split("|"):
                category_set.update(val)

            category_set.add("<PAD>")
            genres2int = {val: ii for ii, val in enumerate(category_set)}

            #将电影类型转成等长数字列表，长度是18
            self.category_map = {val: [genres2int[row] for row in val.split("|")] for ii, val in enumerate(set(movies["category"]))}

            for key in self.category_map:
                for cnt in range(max(genres2int.values()) - len(self.category_map[key])):
                    self.category_map[key].insert(
                        len(self.category_map[key]) + cnt, genres2int["<PAD>"])

        movies["category"] = movies["category"].map(self.category_map)

        #电影name转数字字典
        if need_save_relationship:
            name_set = set()
            for val in movies["name"].str.split():
                name_set.update(val)

            name_set.add("<PAD>")
            name2int = {val: ii for ii, val in enumerate(name_set)}

            #将电影name转成等长数字列表，长度是15
            name_count = 15
            self.name_map = {val: [name2int[row] for row in val.split()] for ii, val in enumerate(set(movies["name"]))}

            for key in self.name_map:
                for cnt in range(name_count - len(self.name_map[key])):
                    self.name_map[key].insert(len(self.name_map[key]) + cnt, name2int["<PAD>"])

        movies["name"] = movies["name"].map(self.name_map)

        self.movies_values = movies.values

        ratings_title = ["uid", "mid", "rating", "ts"]
        ratings = pd.read_table(self.work_dir + "/ratings.dat", sep = "::", header = None, names = ratings_title, engine = "python")
        ratings = ratings.filter(regex = "uid|mid|rating")

        data = pd.merge(pd.merge(ratings, users), movies)

        target_fields = ["rating"]
        features_pd, targets_pd = data.drop(target_fields, axis = 1), data[target_fields]
        
        features = features_pd.values
        targets_values = targets_pd.values

        # save relationship
        if need_save_relationship:
            self.saveRelationship(self.age_map, os.path.join(REPLATIONSHIP_DIR, "age"))
            self.saveRelationship(self.addr_map, os.path.join(REPLATIONSHIP_DIR, "addr"))
            self.saveRelationship(self.mid_map, os.path.join(REPLATIONSHIP_DIR, "mid"))
            self.saveRelationship(self.category_map, os.path.join(REPLATIONSHIP_DIR, "category"))
            self.saveRelationship(self.name_map, os.path.join(REPLATIONSHIP_DIR, "name"))
            self.saveRelationship(self.name_map, os.path.join(REPLATIONSHIP_DIR, "name_year"))
        return features, targets_values

    def saveRelationship(self, dict, file_path):
        path = os.path.abspath(file_path)
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, "w+") as file:
            file.write(str(dict))

    def restoreRelationship(self, file_path):
        path = os.path.abspath(file_path)
        if not os.path.exists(path):
            return None
        with open(path, "r") as file:
            file_content = file.read()
            return ast.literal_eval(file_content)
        return None

    def resoreAllReplationship(self):
        # save relationship
        self.age_map = self.restoreRelationship(os.path.join(REPLATIONSHIP_DIR, "age"))
        self.addr_map = self.restoreRelationship(os.path.join(REPLATIONSHIP_DIR, "addr"))
        self.mid_map = self.restoreRelationship(os.path.join(REPLATIONSHIP_DIR, "mid"))
        self.category_map = self.restoreRelationship(os.path.join(REPLATIONSHIP_DIR, "category"))
        self.name_map = self.restoreRelationship(os.path.join(REPLATIONSHIP_DIR, "name"))
        self.name_map = self.restoreRelationship(os.path.join(REPLATIONSHIP_DIR, "name_year"))
        if self.age_map == None or self.addr_map == None or self.mid_map == None or self.category_map == None or self.name_map == None or self.name_year_map == None:
            return False
        return True

    def transferAge(self, age):
        if self.age_map != None:
            return self.age_map[age]
        return None

    def transferAddr(self, addr):
        if self.addr_map != None:
            return self.addr_map[addr]
        return None

    def transferCategory(self, category):
        if self.category_map != None:
            return self.category_map[category]
        return None

    def transferName(self, name):
        if self.name_map != None:
            return self.name_map[name]
        return None

    def transferMIDToIndex(self, mid):
        if self.mid_map != None:
            return self.mid_map[mid]
        return None

    def getUsersValues(self):
        return self.users_values

    def getMoviesValues(self):
        return self.movies_values

    def getRawUsersValues(self):
        return self.raw_users_values

    def getRawMoviesValues(self):
        return self.raw_movies_values

class Trainer(object):
    def __init__(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

    def get_batches(self, X, Y, batch_size):
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield X[start:end], Y[start:end]

    def train(self, model):
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.mkdir(SAVE_DIR)
        inputer = DataInpter("./ml-1m")
        features, targets_values = inputer.readData()
        losses = { "train": [], "test": [] }
        with tf.Session() as sess:
            # build tf summary file writer
            train_op, inference, loss, gradients, global_step = model.inference(sess.graph)
            uid, sex, age, occ, addr, mid, category, name, target, inference, user_layer, movie_layer, learning_rate, dropout_keep_prob = model.getTensors(sess.graph)
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "train"))
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
                os.mkdir(out_dir)
            print "please watch tensorboard at logdir %s" % out_dir

            with tf.name_scope("accuracy"):
                correct_rate = tf.equal(tf.cast(tf.round(inference), tf.int32), target)
                accuracy = tf.reduce_mean(tf.cast(correct_rate, tf.float32))
                accuracy_summary = tf.summary.scalar("accuracy", accuracy)

            loss_summary = tf.summary.scalar("loss", loss)
            train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])

            train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), sess.graph)
            test_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "test"), sess.graph)

            # begin to train
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for epoch_i in range(self.epoch):
                trainX, testX, trainY, testY = train_test_split(features, targets_values, test_size = 0.2, random_state = 0)
                train_batches = self.get_batches(trainX, trainY, self.batch_size)
                test_batches = self.get_batches(testX, testY, self.batch_size)

                for batch_i in range(len(trainX) // self.batch_size):
                    x, y = next(train_batches)
                    categories = np.zeros([self.batch_size, 18])
                    for i in range(self.batch_size):
                        categories[i] = x.take(7, 1)[i]

                    names = np.zeros([self.batch_size, NAME_WORD_MAX])
                    for i in range(self.batch_size):
                        names[i] = x.take(6, 1)[i]

                    feed = {
                        uid: np.reshape(x.take(0, 1), [self.batch_size, 1]),
                        sex: np.reshape(x.take(2, 1), [self.batch_size, 1]),
                        age: np.reshape(x.take(3, 1), [self.batch_size, 1]),
                        occ: np.reshape(x.take(4, 1), [self.batch_size, 1]),
                        addr: np.reshape(x.take(5, 1), [self.batch_size, 1]),
                        mid: np.reshape(x.take(1, 1), [self.batch_size, 1]),
                        category: categories,
                        name: names,
                        target: np.reshape(y, [self.batch_size, 1]),
                        dropout_keep_prob: DROPOUT_KEEP_PROP,
                        learning_rate: LEARNING_RATE
                    }
                    step, train_loss, summaries, predicts = sess.run([global_step, loss, train_summary_op, train_op], feed_dict = feed)
                    losses['train'].append(train_loss)
                    train_summary_writer.add_summary(summaries, step)

                    if (epoch_i * (len(trainX) // self.batch_size) + batch_i) % 50 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('{}: train epoch {:>3} batch {:>4}/{}   train_loss = {:.3f}'.format( time_str, epoch_i, batch_i, (len(trainX) // self.batch_size), train_loss))
                #使用测试数据的迭代
                for batch_i in range(len(testX) // self.batch_size):
                    x, y = next(test_batches)

                    categories = np.zeros([self.batch_size, 18])
                    for i in range(self.batch_size):
                        categories[i] = x.take(7, 1)[i]

                    names = np.zeros([self.batch_size, NAME_WORD_MAX])
                    for i in range(self.batch_size):
                        names[i] = x.take(6, 1)[i]

                    feed = {
                        uid: np.reshape(x.take(0, 1), [self.batch_size, 1]),
                        sex: np.reshape(x.take(2, 1), [self.batch_size, 1]),
                        age: np.reshape(x.take(3, 1), [self.batch_size, 1]),
                        occ: np.reshape(x.take(4, 1), [self.batch_size, 1]),
                        addr: np.reshape(x.take(5, 1), [self.batch_size, 1]),
                        mid: np.reshape(x.take(1, 1), [self.batch_size, 1]),
                        category: categories,
                        name: names,
                        target: np.reshape(y, [self.batch_size, 1]),
                        dropout_keep_prob: 1,
                        learning_rate: LEARNING_RATE
                    }
                    step, test_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed_dict = feed)
                    #保存测试损失
                    losses['test'].append(test_loss)
                    test_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    if (epoch_i * (len(testX) // self.batch_size) + batch_i) % 50 == 0:
                        print('{}: test epoch {:>3} batch {:>4}/{}   test_loss = {:.3f}'.format(time_str, epoch_i, batch_i, (len(testX) // self.batch_size), test_loss))

                if epoch_i % 5 == 4:
                    saver.save(sess, os.path.join(SAVE_DIR, "saver") + "_%d" % epoch_i)
        print('Model Trained and Saved')

class Predicter(object):
    def __init__(self, model, model_path):
        self.model_path = model_path
        self.model = model

    def loadGraphFromFile(self, sess, file_path):
        graph_loader = tf.train.import_meta_graph(file_path + ".meta")
        graph_loader.restore(sess, file_path)

    def ratingMovie(self, user_tensors, movie_tensors):
        categories = np.zeros([1, 18])
        categories[0] = movie_tensors[2]

        names = np.zeros([1, NAME_WORD_MAX])
        names[0] = movie_tensors[1]
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            self.loadGraphFromFile(sess, self.model_path)
            uid, sex, age, occ, addr, mid, category, name, target, inference, user_layer, movie_layer, lr, dropout_keep_prob = self.model.getTensors(graph)
            feed = {
                uid: np.reshape(user_tensors[0], [1, 1]),
                sex: np.reshape(user_tensors[1], [1, 1]),
                age: np.reshape(user_tensors[2], [1, 1]),
                occ: np.reshape(user_tensors[3], [1, 1]),
                addr: np.reshape(user_tensors[4], [1, 1]),
                mid: np.reshape(movie_tensors[0], [1, 1]),
                name: names,
                category: categories,
                dropout_keep_prob: 1,
            }
            inference_val = sess.run([inference], feed)
            return inference_val

    def genreateMovieMatrix(self, movies_values):
        movie_matrics = []
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            self.loadGraphFromFile(sess, self.model_path)
            uid, sex, age, occ, addr, mid, category, name, target, inference, user_layer, movie_layer, lr, dropout_keep_prob = self.model.getTensors(graph)

            for movie_info in movies_values:
                categories = np.zeros([1, 18])
                categories[0] = movie_info.take(2)

                names = np.zeros([1, NAME_WORD_MAX])
                names[0] = movie_info.take(1)

                feed = {
                    mid: np.reshape(movie_info.take(0), [1, 1]),
                    category: categories,
                    name: names,
                    dropout_keep_prob: 1,
                }

                movie_fc_layer = sess.run([movie_layer], feed_dict = feed)
                movie_matrics.append(movie_fc_layer)
        movie_matrics = np.reshape(movie_matrics, [-1, USER_OUTPUT_DIMENSION])
        return movie_matrics


    def generateUserMatrix(self, users_values):
        user_matrics = []
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            self.loadGraphFromFile(sess, self.model_path)
            uid, sex, age, occ, addr, mid, category, name, target, inference, user_layer, movie_layer, lr, dropout_keep_prob = self.model.getTensors(graph)

            for user_info in users_values:
                feed = {
                    uid: np.reshape(user_info.take(0), [1, 1]),
                    sex: np.reshape(user_info.take(1), [1, 1]),
                    age: np.reshape(user_info.take(2), [1, 1]),
                    occ: np.reshape(user_info.take(3), [1, 1]),
                    addr: np.reshape(user_info.take(4), [1, 1]),
                    dropout_keep_prob: 1,
                }

                user_fc_layer = sess.run([user_layer], feed)
                user_matrics.append(user_fc_layer)
        user_matrics = np.reshape(user_matrics, [-1, USER_OUTPUT_DIMENSION])
        return user_matrics

    def recommendMovieByMovie(self, movie_matrics, mid, inputer, count):
        #根据mid获取电影的index
        mid_index = inputer.transferMIDToIndex(mid)
        # 做一个类似Normalization的操作，把电影特征值分布到合理范围内
        norm_movie_matrics = np.sqrt(np.add.reduce(np.square(movie_matrics)))
        normalized_movie_matrics = np.divide(movie_matrics, norm_movie_matrics)

        # 推荐相似的的电影
        probs_embeddings = (movie_matrics[mid_index]).reshape([1, USER_OUTPUT_DIMENSION])
        # 当前电影特征与所有电影特征做矩阵乘法
        probs_similarity = np.dot(probs_embeddings, normalized_movie_matrics.T)

        current_movie = inputer.getRawMoviesValues()[mid_index]
        print("您看的电影是：{}, 类型是：{}".format(current_movie[1], current_movie[2]))
        print("以下是相似的电影：")
        p = np.squeeze(probs_similarity)
        p[np.argsort(p)[:-count]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 10:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            recommend_movie = inputer.getRawMoviesValues()[val]
            print("电影是：{}, 类型是：{}".format(recommend_movie[1], recommend_movie[2]))

        return results
    
    def recommendMovieByUser(self, user_matrics, movie_matrics, uid, inputer, count):
        uid_index = uid - 1
        #猜你喜欢的电影
        probs_embeddings = (user_matrics[uid_index]).reshape([1, USER_OUTPUT_DIMENSION])
        probs_similarity = np.dot(probs_embeddings, movie_matrics.T)
        print("以下是给您的推荐：")
        p = np.squeeze(probs_similarity)
        p[np.argsort(p)[:-count]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 10:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            recommend_movie = inputer.getRawMoviesValues()[val]
            print("电影是：{}, 类型是：{}".format(
                recommend_movie[1], recommend_movie[2]))

        return results

def predictMain():
    model = Model(False)
    predicter = Predicter(model, os.path.abspath("./train_data/saver_19"))

    inputer = DataInpter("/Users/sjjwind/Downloads/ml-1m")
    inputs, labels = inputer.readData()

    # 预测用户的电影评分
    predictRating = False
    if predictRating:
        uid, mid = (234, 1401)
        user_tensor = inputer.getUsersValues()[uid][:]
        movie_tensor = inputer.getMoviesValues()[mid][:]

        print 'user info:', user_tensor
        print 'movie info:', movie_tensor
        print '预测电影评分:', predicter.ratingMovie(user_tensor, movie_tensor)

    # 给用户推荐电影
    mid = 245
    movie_matrics = predicter.genreateMovieMatrix(inputer.getMoviesValues())
    predicter.recommendMovieByMovie(movie_matrics, mid, inputer, 30)

    uid = 245
    user_matrics = predicter.generateUserMatrix(inputer.getUsersValues())
    predicter.recommendMovieByUser(user_matrics, movie_matrics, uid, inputer, 30)


def trainMain():
    model = Model(False)
    trainer = Trainer(EPOCHS_COUNT, BATCH_SIZE)
    trainer.train(model)


def main():
    trainMain()
    predictMain()

if __name__ == '__main__':
    main()

