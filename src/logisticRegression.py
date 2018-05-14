#!virtualenv tensorflow python3
# -*-coding: utf-8-*-
""""""
__title__ = ''
__author__ = 'zxx'
__mtime__ = '18-5-10'

import tensorflow as tf
import os

TRAIN_STEPS = 10000
LEARNING_RATE = 0.001

COLUMNS = ["passenger_id", "survived", "pclass", "name", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin",
           "embarked"]
RECORD_DEFAULTS = [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]]
BATCH_SIZE = 10

W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(x):
    # 先对x做线性拟合，然后使用sigmod函数
    return tf.sigmoid(tf.matmul(x, W) + b)


def loss(x, y):
    y_predict = inference(x)
    # 计算交叉熵
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=y_predict))


def inputs():
    survived, pclass, sex, age = read_csv(BATCH_SIZE, "train2.csv", [[0.0], [0], [""], [0.0]])

    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [BATCH_SIZE, 1])

    return features, survived


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(queue=filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    #TODO 总是提示randomShuffleQuere方法outOfRange，不会调试
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def train(loss):
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=loss)


def evaluate(sess, x, y):
    predicted = tf.cast(inference(x) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(TRAIN_STEPS):
        sess.run([train_op])
        if step % 100 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
