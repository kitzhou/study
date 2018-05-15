#!virtualenv tensorflow python3
#-*-coding: utf-8-*-
"""softmax多值分类"""
__title__ = ''
__author__ = 'zxx'
__mtime__ = '18-5-15'

import tensorflow as tf
import config

TRAIN_STEPS = 1000
LEARNING_RATE = 0.01

RECORD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [""]]
BATCH_SIZE = 20

W = tf.Variable(tf.zeros([4,3]), name="weights")
b = tf.Variable(tf.zeros([3]), name="bias")


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([config.getProjectRoot() + "/Iris/" + file_name])

    reader = tf.TextLineReader()
    key, value = reader.read(queue=filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label= read_csv(BATCH_SIZE, "iris.data", RECORD_DEFAULTS)

    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack(
        [tf.equal(label, ["Iris-setosa"]), tf.equal(label, ["Iris-versicolor"]),
         tf.equal(label, ["Iris-virginica"])])),0))
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

    return features, label_number


def inference(x):
    """先对x做线性拟合,再用softmax分类"""
    return tf.nn.softmax(tf.matmul(x, W) + b)


def loss(x, y):
    y_predict = inference(x)
    # 计算交叉熵
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=y_predict))


def train(loss):
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=loss)


def evaluate(sess, x, y):
    predicted = tf.cast(tf.argmax(inference(x),1),tf.int32)

    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    X, Y = inputs()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(TRAIN_STEPS):
        sess.run([train(loss(X, Y))])

        if step % 10 == 0:
            print("loss: ", sess.run([loss(X, Y)]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()