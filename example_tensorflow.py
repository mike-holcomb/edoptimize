import tensorflow as tf
import numpy as np
import heapq


def make_model(name, input_size=2, output_size=1):
    model_layers = {}
    with tf.name_scope(name):
        x = tf.placeholder(tf.float32,shape=(None, input_size))
        model_layers["x"] = x

        W = tf.Variable(initial_value=tf.random.normal(shape=[input_size,output_size]), dtype=tf.float32)
        model_layers["W"] = W

        b = tf.Variable(initial_value=tf.random.normal(shape=[output_size]), dtype=tf.float32)
        model_layers["b"] = b

        o = tf.nn.sigmoid(tf.add(tf.matmul(x,W),b))
        model_layers["o"] = o

        y = tf.placeholder(tf.float32,shape=(None, output_size))
        model_layers["y"] = y

        loss = tf.losses.mean_squared_error(y,o)
        model_layers["loss"] = loss

    return model_layers


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]],dtype=np.float32)

    Y = np.array([[0],
                  [1],
                  [1],
                  [1]],dtype=np.float32)

    N = 10
    k = N // 2

    with tf.Session() as sess:
        models = [None] * N
        losses = {}
        input_size = 2
        output_size = 1

        averages = [ tf.Variable(initial_value=tf.zeros(shape=[input_size,output_size]), dtype=tf.float32),
                     tf.Variable(initial_value=tf.zeros(shape=[output_size]), dtype=tf.float32) ]
        stddev = [ tf.Variable(initial_value=tf.zeros(shape=[input_size,output_size]), dtype=tf.float32),
                     tf.Variable(initial_value=tf.zeros(shape=[output_size]), dtype=tf.float32) ]

        for i in range(N):
            models[i] = make_model("model_{}".format(i))

        sess.run([tf.global_variables_initializer()])

        for i in range(N):
            model = models[i]
            losses[i] = sess.run([ model.get("loss") ],
                                 feed_dict={ model.get("x"):X, model.get("y"):Y })

        best_nets = heapq.nsmallest(k, losses,losses.get)

        estimate_mean

    print(losses)
    print(best_nets)


if __name__ == "__main__":
    main()
