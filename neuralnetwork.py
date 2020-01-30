import pandas as pd
import tensorflow as tf
import numpy as np
from helper import logging
import os

import warnings

warnings.filterwarnings('ignore')

seed = 42
ENCODER_BASE_NAME = 'stacked_autoencoder'
PATH_SAVED_MODEL = 'saved_model'
PATH_OUTPUT = 'output_data'


def fullyConnected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        lambda_l2 = 0.001
        out = tf.layers.dense(input, output_size, name=name, activation=tf.nn.selu,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                mode='FAN_IN'),
                              bias_initializer=tf.initializers.zeros(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_l2))
        return out


def calibrate_stacked_autoencoder(n_epoch, input_train, input_test, n_bottleneck, noise_stddev_, batch_size=50):
    reset_graph()
    encoder_name = ENCODER_BASE_NAME + "_noiseStdev" + str(noise_stddev_) + "_bl" + str(n_bottleneck)

    learning_rate_epoch = 0.001
    learning_rate_decay = 0.9
    learning_rate_decay_iter = 1000

    n_input_neurons = input_train.shape[1]
    logging(str(n_input_neurons) + ' input neurons.')

    with tf.variable_scope("placeholders"):
        X = tf.placeholder(tf.float32, shape=[None, n_input_neurons], name="inputs")
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
        noise_stddev = tf.placeholder(dtype=tf.float32, shape=[], name="noise_stddev")

    X_noisy_input = tf.cond(noise_stddev <= 0.0, lambda: X, \
                            lambda: tf.add(X, tf.random_normal(shape=tf.shape(X), mean=0.0, stddev=noise_stddev_)))

    with tf.variable_scope("dense_layers"):
        hidden_layer = fullyConnected(X_noisy_input, name='encoder_1', output_size=int(n_input_neurons * 0.75))
        logging(str(hidden_layer.name) + '.shape:' + str(hidden_layer.shape))

        hidden_layer = fullyConnected(hidden_layer, name='encoder_2', output_size=int(n_input_neurons * 0.5))
        logging(str(hidden_layer.name) + '.shape:' + str(hidden_layer.shape))

        hidden_layer = fullyConnected(hidden_layer, name='codings', output_size=int(n_bottleneck))
        logging(str(hidden_layer.name) + '.shape:' + str(hidden_layer.shape))

        hidden_layer = fullyConnected(hidden_layer, name='decoder_1', output_size=int(n_input_neurons * 0.5))
        logging(str(hidden_layer.name) + '.shape:' + str(hidden_layer.shape))

        hidden_layer = fullyConnected(hidden_layer, name='decoder_2', output_size=int(n_input_neurons * 0.75))
        logging(str(hidden_layer.name) + '.shape:' + str(hidden_layer.shape))

        outputs = tf.layers.dense(hidden_layer, n_input_neurons, name="outputs")
        logging(str(outputs.name) + '.shape:' + str(outputs.shape))

    with tf.variable_scope("operations"):
        reconstruction_loss = tf.losses.huber_loss(predictions=outputs, labels=X)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(reconstruction_loss)
        init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=None)
    loss_train_test_epoch = []
    with tf.Session() as sess:
        init.run()
        for i_epoch in range(n_epoch + 1):
            if (i_epoch % learning_rate_decay_iter == 0 and i_epoch > 1):
                learning_rate_epoch = learning_rate_epoch * learning_rate_decay
                logging('set new learning_rate:' + str(learning_rate_epoch))
            for i, batch in enumerate(
                    iterate_minibatches(inputs=[input_train], targets=[], weights=[], batch_size=batch_size,
                                        shuffle=True)):
                x_batch, y_batch, calib_weights_batch = batch
                training_op.run(
                    feed_dict={X: x_batch[0], learning_rate: learning_rate_epoch, noise_stddev: noise_stddev_})

            if (i_epoch % 100 == 0 and i_epoch > 1):
                loss_train = reconstruction_loss.eval(
                    feed_dict={X: input_train, learning_rate: learning_rate_epoch, noise_stddev: 0})
                loss_test = reconstruction_loss.eval(
                    feed_dict={X: input_test, learning_rate: learning_rate_epoch, noise_stddev: 0})
                loss_train_test_epoch.append([i_epoch, loss_train, loss_test])
                logging('save model, epoch:' + str(i_epoch) + ', loss_train:' + str(
                    loss_train) + ', loss_test:' + str(loss_test))
                saver.save(sess, os.path.join(os.getcwd(), PATH_SAVED_MODEL, encoder_name + ".ckpt"))
                df_loss_train_test_epoch = pd.DataFrame(data=loss_train_test_epoch,
                                                        columns=['iEpoch', 'loss_train', 'loss_test'])
                df_loss_train_test_epoch.to_csv(os.path.join(os.getcwd(), PATH_OUTPUT, encoder_name + ".csv"), sep=';')


def reset_graph():
    global seed
    seed = 42
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def train_dev_test_split(inputs, split_ratio, shuffle=True):
    nb_examples = inputs[0].shape[0]
    assert split_ratio[0] + split_ratio[1] <= 1.0
    if shuffle:
        indices = np.arange(nb_examples)
        np.random.seed(42)
        np.random.shuffle(indices)
    else:
        indices = np.arange(0, nb_examples)
    idx_train = indices[0:int(nb_examples * split_ratio[0])]
    idx_dev = indices[min(int(nb_examples * split_ratio[0]) + 1, nb_examples - 1):  int(
        nb_examples * (split_ratio[0] + split_ratio[1]))]
    idx_test = indices[min(int(nb_examples * (split_ratio[0] + split_ratio[1])) + 1, nb_examples - 1):  nb_examples - 1]

    inputs_train = []
    inputs_dev = []
    inputs_test = []
    for input_instance in inputs:
        if input_instance is None:
            inputs_train.append(None)
            inputs_dev.append(None)
            inputs_test.append(None)
        else:
            assert input_instance.shape[0] == nb_examples
            inputs_train.append(input_instance[idx_train])
            inputs_dev.append(input_instance[idx_dev])
            inputs_test.append(input_instance[idx_test])

    split = {}
    split["train"] = [inputs_train]
    split["dev"] = [inputs_dev]
    split["test"] = [inputs_test]

    return split


def get_train_test_index(df, split_ratio):
    global seed
    seed = 42
    trainTestIndex = train_dev_test_split([df.index], split_ratio)
    return trainTestIndex['train'][0][0], trainTestIndex['test'][0][0]


def iterate_minibatches(inputs, targets, weights, batch_size, shuffle=False):
    global seed
    nb_examples = inputs[0].shape[0]
    for input_element in inputs:
        assert input_element.shape[0] == nb_examples, "invalid inputs"
    if targets is not None:
        for target_element in targets:
            if target_element is not None:
                assert target_element.shape[0] == nb_examples, "invalid inputs"
    for weight_element in weights:
        assert weight_element.shape[0] == nb_examples, "invalid inputs"

    if shuffle:
        indices = np.arange(nb_examples)
        seed += 1
        np.random.seed(seed)
        np.random.shuffle(indices)
    for start_idx in range(0, nb_examples, batch_size):
        end_idx = min(start_idx + batch_size, nb_examples)
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = range(start_idx, end_idx, 1)
        minibatch_inputs = [input_element[excerpt] for input_element in inputs]
        minibatch_outputs = [target_element[excerpt] for target_element in targets if target_element is not None]
        minibatch_weights = [weight_element[excerpt] for weight_element in weights]
        if len(minibatch_outputs) == 0:
            minibatch_outputs = [None]
        yield minibatch_inputs, minibatch_outputs, minibatch_weights


def get_prediction_from_model(dfInput, noise_stddev, n_bottleneck, index_for_statistics):
    reset_graph()
    encoder_name = ENCODER_BASE_NAME + "_noiseStdev" + str(noise_stddev) + "_bl" + str(n_bottleneck)
    saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), PATH_SAVED_MODEL, encoder_name + ".ckpt.meta"))
    X = tf.get_default_graph().get_tensor_by_name(name="placeholders/inputs:0")
    outputs = tf.get_default_graph().get_tensor_by_name(name="dense_layers/outputs/BiasAdd:0")
    learning_rate = tf.get_default_graph().get_tensor_by_name(name="placeholders/learning_rate:0")
    noise_stddev = tf.get_default_graph().get_tensor_by_name(name="placeholders/noise_stddev:0")

    with tf.Session() as sess:
        saver.restore(sess, (os.path.join(os.getcwd(), PATH_SAVED_MODEL, encoder_name + '.ckpt')))
        outputs = outputs.eval(feed_dict={X: dfInput['data'].values, learning_rate: 1.0, noise_stddev: 0})

    dfResults = {}
    dfResults['data'] = pd.DataFrame(data=outputs, columns=dfInput['data'].columns, index=dfInput['data'].index)
    dfResults['RL'] = pd.DataFrame(data=np.sqrt(np.sum((dfResults['data'] - dfInput['data']) ** 2, axis=1)),
                                   columns=['RL'])
    dfResults['RL'] = dfResults['RL'].sort_index()
    dfResults['RLmean'] = np.mean(dfResults['RL'].loc[index_for_statistics])
    dfResults['RLstd'] = np.std(dfResults['RL'].loc[index_for_statistics])
    dfResults['zValue'] = pd.DataFrame(data=abs(dfResults['RL'] - dfResults['RLmean']) / dfResults['RLstd']).rename(
        columns={'RL': 'zValue'})
    return encoder_name, dfResults
