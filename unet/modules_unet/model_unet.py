#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models

# UNET Model 1


def model(nbr, x, y, dim_inp=3):
    entry = layers.Input(shape=(x, y, dim_inp), dtype='float32')

    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(entry)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result1 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result1)

    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result2 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result2)

    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result3 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result3)

    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result4 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result4)

    result = layers.Conv2D(8*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.UpSampling2D()(result)
    result = tf.concat([result, result4], axis=3)

    result = layers.Conv2D(8*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.UpSampling2D()(result)
    result = tf.concat([result, result3], axis=3)

    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.UpSampling2D()(result)
    result = tf.concat([result, result2], axis=3)

    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.UpSampling2D()(result)
    result = tf.concat([result, result1], axis=3)

    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    output = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(result)

    model = models.Model(inputs=entry, outputs=output)
    return model
