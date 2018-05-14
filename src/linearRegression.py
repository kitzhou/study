import tensorflow as tf

TRAIN_STEPS = 10000
LEARNING_RATE = 0.0000001

W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    Y_predict = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predict))


def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  (59, 46), [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374,
                         308, 220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=total_loss)


def evaluate(sess, X, Y):
    print(sess.run(inference([[80.0, 25.0]]))) # ~303
    print(sess.run(inference([[65.0, 25.0]]))) # ~256


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
