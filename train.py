# -*- encoding:utf-8 -*-

import input
import flags as fl
import model
import tensorflow as tf
import time, datetime
import os

class ModelTrainer(object):
    def __init__(self, model, batchSize):
        self.model = model
        self.inputer = input.InputManager(batchSize)
        self.saveDir = '/tmp/tf_saver/data_tmp'
        try:
            os.mkdir(os.path.pardir(self.saveDir))
        except Exception, e:
            pass
        self.batchSize = batchSize

    def show_shape(tensor):
          print ('%s.get_shape():' % tensor.name), tensor.get_shape()

    def train(self):
        num_epochs = 5
        losses = {'train': [], 'test': []}
        train_op, loss, gradients, global_step = self.model.inference()
        with tf.Session() as sess:
            grad_summaries = []
            for g, v in gradients:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name.replace(':', '_')), g)
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(
                os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)

            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)

            # Inference summaries
            inference_summary_op = tf.summary.merge([loss_summary])
            inference_summary_dir = os.path.join(
                out_dir, "summaries", "inference")
            inference_summary_writer = tf.summary.FileWriter(
                inference_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for epoch_i in range(num_epochs):
                index = 0
                print('new epoch:', epoch_i)
                while self.inputer.hasNext():
                    index = index + 1
                    if index % 100 == 0:
                        print 'epoch: %d, index: %d' % (epoch_i, index)
                    uids, sexs, ages, occs = self.inputer.getNextUserInfos()
                    mids, genreses, names = self.inputer.getNextMovieInfos()
                    targets = self.inputer.getNextTargets()
                    self.inputer.next()

                    feed = {
                        'uid:0': uids,
                        'sex:0': sexs,
                        'age:0': ages,
                        'occ:0': occs,
                        'mid:0': mids,
                        'category:0': genreses,
                        'movie_name:0': names,
                        'targets:0': targets,
                        'dropout_keep_prob:0': fl.dropout_keep,
                        'learning_rate:0': fl.learning_rate
                    }

                    step, train_loss, summaries, _ = sess.run(
                        [global_step, loss, train_summary_op, train_op], feed)
                    losses['train'].append(train_loss)
                    train_summary_writer.add_summary(summaries, step)

                testUids, testSexs, testAges, testOccs = self.inputer.getTestUserInfos()
                testMids, testGenreses, testNames = self.inputer.getTestMovieInfos()
                testTargets = self.inputer.getTestTargets()
                print('begin to test')
                for batch_i in range(len(testTargets) // self.batchSize):
                    currentIndex = batch_i * self.batchSize
                    currentIndexEnd = currentIndex + self.batchSize
                    testFeed = {
                        'uid:0': testUids[currentIndex:currentIndexEnd],
                        'sex:0': testSexs[currentIndex:currentIndexEnd],
                        'age:0': testAges[currentIndex:currentIndexEnd],
                        'occ:0': testOccs[currentIndex:currentIndexEnd],
                        'mid:0': testMids[currentIndex:currentIndexEnd],
                        'category:0': testGenreses[currentIndex:currentIndexEnd],
                        'movie_name:0': testNames[currentIndex:currentIndexEnd],
                        'targets:0': testTargets[currentIndex:currentIndexEnd],
                        'dropout_keep_prob:0': 1,
                        'learning_rate:0': fl.learning_rate
                    }

                    step, test_loss, summaries = sess.run(
                        [global_step, loss, inference_summary_op], testFeed)  # cost

                    losses['test'].append(test_loss)
                    inference_summary_writer.add_summary(summaries, step)  #

                    time_str = datetime.datetime.now().isoformat()
                    if batch_i % 1000 == 0:
                        print('{}: Epoch {:>3}  test_loss = {:.3f}'.format(
                            time_str, epoch_i, test_loss))
                self.inputer.nextEpoch()

            saver.save(sess, self.saveDir)
        print('Model Trained and Saved')

def test():
    modelTrainer = ModelTrainer(model.Model(96, 50), 96)
    modelTrainer.train()

if __name__ == '__main__':
    test()
