# -*- encoding:utf-8 -*-

import tensorflow as tf

FLAGS = tf.flags.FLAGS

embed_dimension = 32

# uid 最多13个二进制位即可表示
uid_dimension = 13

# sex 最多1个二进制位即可表示
sex_dimension = 1

# age 最多3个二进制位即可表示
age_dimension = 3

# occ 最多5个二进制位即可表示
occ_dimension = 5

mid_dimension = 13
title_dimension = 6000
category_dimension = 19

filter_num = 8
sentences_size = 15

learning_rate = 0.001

dropout_keep = 0.5
