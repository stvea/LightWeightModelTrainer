import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
import time



def batch_norm(training):
    return layers.BatchNormalization(axis=3, momentum=0.99,
                                     epsilon=0.001, center=True,
                                     scale=True, trainable=training, fused=True)


class ShufflenetV2(tf.keras.Model):
    def __init__(self,num_classes, training):
        super(ShufflenetV2, self).__init__()
        self._set_inputs(tf.TensorSpec([None, 224, 224, 3], tf.float32, name="inputs"))
        self.training = training
        self.num_classes = num_classes

        self.conv1 = layers.Conv2D(24, kernel_size=3, strides=2, padding='SAME')
        self.bn1 = batch_norm(self.training)
        self.act1 = layers.Activation("relu")
        self.maxpool1 = layers.MaxPooling2D((3, 3), (2, 2), padding='SAME')

        self.block1 = ShuffleBlock(num_units=4, in_channels=24, out_channels=116)
        self.block2 = ShuffleBlock(num_units=8, in_channels=116)
        self.block3 = ShuffleBlock(num_units=4, in_channels=232)

        self.globalavgpool = layers.GlobalAveragePooling2D()
        self.dropout1 = layers.Dropout(rate=0.7)
        self.dense1 = layers.Dense(num_classes)

        self.conv5 = layers.Conv2D(1024, kernel_size=1, strides=1, padding='SAME')
        self.bn5 = batch_norm(self.training)
        self.act5 = layers.Activation("softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.globalavgpool(x)
        x = self.dropout1(x, training=self.training)
        x = self.dense1(x)

        return x

    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     x, y = data
    #
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #
    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}


class ShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, in_channels, out_channels=None, training=True):
        super(ShuffleBlock, self).__init__()

        self.training = training
        self.num_units = num_units
        self.in_channels = in_channels
        self.out_channels = 2 * self.in_channels if out_channels is None else out_channels

        self.all_basic_uint = []
        for j in range(2, self.num_units + 1):
            self.all_basic_uint.append(BasicUnit(in_channels=self.out_channels // 2))

        self.conv1 = layers.Conv2D(self.in_channels, kernel_size=1, strides=1, padding='SAME')
        self.bn1 = batch_norm(self.training)
        self.act1 = layers.Activation("relu")

        self.dwconv2 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME')
        self.bn2 = batch_norm(self.training)

        self.conv3 = layers.Conv2D(self.out_channels // 2, kernel_size=1, strides=1, padding='SAME')
        self.bn3 = batch_norm(self.training)
        self.act3 = layers.Activation("relu")

        self.dwconv4 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME')
        self.bn4 = batch_norm(self.training)

        self.conv5 = layers.Conv2D(self.out_channels // 2, kernel_size=1, strides=1, padding='SAME')
        self.bn5 = batch_norm(self.training)
        self.act5 = layers.Activation("relu")

    def shuffle_xy(self, x, y):
        batch_size, height, width, channels = x.shape[:]
        depth = channels

        z = tf.stack([x, y], axis=3)
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [-1, height, width, 2 * depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        y = self.dwconv2(y)
        y = self.bn2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.act3(y)

        x = self.dwconv4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        basic_uint_count = 0
        for j in range(2, self.num_units + 1):
            x, y = self.shuffle_xy(x, y)
            x = self.all_basic_uint[basic_uint_count](x)
            basic_uint_count += 1

        x = layers.concatenate([x, y])
        return x


class BasicUnit(tf.keras.layers.Layer):
    def __init__(self, in_channels=10, training=True):
        super(BasicUnit, self).__init__()
        self.in_channels = in_channels

        self.training = training
        self.conv1 = layers.Conv2D(self.in_channels, kernel_size=1, strides=1, padding='SAME')
        self.bn1 = batch_norm(self.training)
        self.act1 = layers.Activation("relu")

        self.dwconv2 = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='SAME')
        self.bn2 = batch_norm(self.training)

        self.conv3 = layers.Conv2D(self.in_channels, kernel_size=1, strides=1, padding='SAME')
        self.bn3 = batch_norm(self.training)
        self.act3 = layers.Activation("relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dwconv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x

# class ShufflenetV2(tf.keras.Model):
#     def __init__(self):
#         super(ShufflenetV2, self).__init__()
#         first_stage_channels = 144
#         self.conv1 = tf.keras.layers.Conv2D(filters=24,
#                                    kernel_size=3,
#                                    strides=2,
#                                    padding='same')
#         self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
#         self.stage1 = self.stage(x, first_stage_channels, n=3)
#     def channel_shuffle(self, inputs, num_groups):
#         n, h, w, c = inputs.shape
#         x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
#         x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
#         output = tf.reshape(x_transposed, [-1, h, w, c])
#         return output
#
#     def conv(self, inputs, filters, kernel_size, strides=1):
#         x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(inputs)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Activation('relu')(x)
#         return x
#
#     def depthwise_conv_bn(self, inputs, kernel_size, strides=1):
#         x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
#                                             strides=strides,
#                                             padding='same')(inputs)
#         x = tf.keras.layers.BatchNormalization()(x)
#
#         return x
#
#     def ShuffleNetUnitA(self, inputs, out_channels):
#         shortcut, x = tf.split(inputs, 2, axis=-1)
#
#         x = self.conv(inputs, out_channels // 2, kernel_size=1, strides=1)
#         x = self.depthwise_conv_bn(x, kernel_size=3, strides=1)
#         x = self.conv(x, out_channels // 2, kernel_size=1, strides=1)
#
#         x = tf.concat([shortcut, x], axis=-1)
#         x = self.channel_shuffle(x, 2)
#
#         return x
#
#     def ShuffleNetUnitB(self, inputs, out_channels):
#         shortcut = inputs
#
#         in_channels = inputs.shape[-1]
#
#         x = self.conv(inputs, out_channels // 2, kernel_size=1, strides=1)
#         x = self.depthwise_conv_bn(x, kernel_size=3, strides=2)
#         x = self.conv(x, out_channels - in_channels, kernel_size=1, strides=1)
#
#         shortcut = self.depthwise_conv_bn(shortcut, kernel_size=3, strides=2)
#         shortcut = self.conv(shortcut, in_channels, kernel_size=1, strides=1)
#
#         output = tf.concat([shortcut, x], axis=-1)
#         output = self.channel_shuffle(output, 2)
#
#         return output
#
#     def stage(self, inputs, out_channels, n):
#         x = self.ShuffleNetUnitB(inputs, out_channels)
#
#         for _ in range(n):
#             x = self.ShuffleNetUnitA(x, out_channels)
#
#         return x
#
#     def call(self, inputs):
#         first_stage_channels = 144
#         x = tf.keras.layers.Conv2D(filters=24,
#                                    kernel_size=3,
#                                    strides=2,
#                                    padding='same')(inputs)
#         x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
#
#         x = self.stage(x, first_stage_channels, n=3)
#         x = self.stage(x, first_stage_channels * 2, n=7)
#         x = self.stage(x, first_stage_channels * 4, n=3)
#
#         x = tf.keras.layers.Conv2D(filters=1024,
#                                    kernel_size=1,
#                                    strides=1,
#                                    padding='same')(x)
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         x = tf.keras.layers.Dense(30,activation='softmax')(x)
#
#         return x
