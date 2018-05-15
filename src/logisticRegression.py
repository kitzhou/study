#!virtualenv tensorflow python3
# -*-coding: utf-8-*-
"""对数几率回归，sigmod函数的0-1突变应用，使用交叉熵计算损失"""
__title__ = ''
__author__ = 'zxx'
__mtime__ = '18-5-10'

import tensorflow as tf
import config

TRAIN_STEPS = 1000
LEARNING_RATE = 0.01

COLUMNS = ["passenger_id", "survived", "pclass", "name", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin",
           "embarked"]
RECORD_DEFAULTS = [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]]
BATCH_SIZE = 10

W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([config.getProjectRoot() + "/titanicData/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(queue=filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    results = read_csv(BATCH_SIZE, "train.csv", RECORD_DEFAULTS)

    is_first_class = tf.to_float(tf.equal(results[2], [1]))
    is_second_class = tf.to_float(tf.equal(results[2], [2]))
    is_third_class = tf.to_float(tf.equal(results[2], [3]))

    gender = tf.to_float(tf.equal(results[4], ["female"]))

    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, results[5]]))
    survived = tf.reshape(results[1], [BATCH_SIZE, 1])

    return features, survived


def inference(x):
    """先对x做线性拟合，然后使用sigmod函数"""
    return tf.sigmoid(tf.matmul(x, W) + b)


def loss(x, y):
    y_predict = inference(x)
    # 计算交叉熵
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_predict))


def train(loss):
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=loss)


def evaluate(sess, x, y):
    predicted = tf.cast(inference(x) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    X, Y = inputs()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(TRAIN_STEPS):
        sess.run([train(loss(X, Y))])

        if step % 10 == 0:
            print("loss: ",step, sess.run([loss(X, Y)]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
