import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings('ignore')

ENCODER_BASE_NAME = 'stacked_autoencoder_tf2'
PATH_SAVED_MODEL = 'saved_model'
PATH_OUTPUT = 'output_data'


def build_autoencoder_model(n_bottleneck, n_input_size, noise_stddev):
    encoder = keras.models.Sequential()
    encoder.add(layers.InputLayer(n_input_size))
    if noise_stddev > 0:
        encoder.add(layers.GaussianNoise(noise_stddev))
    encoder.add(layers.Dense(int(n_input_size * 0.75), activation='selu',
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in'),
                             bias_initializer=tf.initializers.zeros()))
    encoder.add(layers.Dense(int(n_input_size * 0.5), activation='selu',
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in'),
                             bias_initializer=tf.initializers.zeros()))
    encoder.add(layers.Dense(n_bottleneck, activation='selu',
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in'),
                             bias_initializer=tf.initializers.zeros()))

    decoder = keras.models.Sequential()
    decoder.add(layers.InputLayer(n_bottleneck))
    decoder.add(layers.Dense(int(n_input_size * 0.5), activation='selu',
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in'),
                             bias_initializer=tf.initializers.zeros()))
    decoder.add(layers.Dense(int(n_input_size * 0.75), activation='selu',
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in'),
                             bias_initializer=tf.initializers.zeros()))
    decoder.add(layers.Dense(n_input_size, activation=None))

    return encoder, decoder


def calibrate_stacked_autoencoder(n_epoch, input_train, input_test, n_bottleneck, noise_stddev_):
    n_input_size = input_train.shape[1]
    encoder_name = ENCODER_BASE_NAME + "_noiseStdev" + str(noise_stddev_) + "_bl" + str(n_bottleneck)
    encoder, decoder = build_autoencoder_model(n_bottleneck, n_input_size, noise_stddev_)

    input_ = layers.Input(n_input_size)
    code = encoder(input_)
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=input_, outputs=reconstruction)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.Huber()
    autoencoder.compile(optimizer=optimizer, loss=loss)

    log_dir = os.path.join('tensorboard', encoder_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

    autoencoder.fit(x=input_train, y=input_train, batch_size=50, epochs=n_epoch,
                    validation_data=[input_test, input_test], verbose=1,
                    callbacks=[tensorboard_callback])
    autoencoder.save(os.path.join(PATH_SAVED_MODEL, encoder_name + '.h5'), save_format='h5')


def get_prediction_from_model(dfInput, noise_stddev, n_bottleneck, index_for_statistics):
    encoder_name = ENCODER_BASE_NAME + "_noiseStdev" + str(noise_stddev) + "_bl" + str(n_bottleneck)
    autoencoder = tf.keras.models.load_model(os.path.join(PATH_SAVED_MODEL, encoder_name + '.h5'))
    outputs = autoencoder.predict(dfInput['data'].values)

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
