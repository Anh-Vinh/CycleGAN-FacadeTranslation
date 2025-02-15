import tensorflow as tf
from tensorflow.keras import layers, models

# Thay thế Instance Normalization tự định nghĩa bằng BatchNormalization vì TF không hỗ trợ (PyTorch có)
def convolutional_block(x, filters, kernel_size=3, strides=2, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    x = layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, filters, kernel_size=3, strides=1, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

# Fractionally-strided convolution
def upsample_block(x, filters, kernel_size=3, strides=2, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(filters, kernel_size, strides, padding, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_generator(input_shape, conv_num, res_num, upsample_num):
    filters = 64
    inputs = layers.Input(shape=input_shape)
    
    x = convolutional_block(x=inputs, filters=filters, kernel_size=7, strides=1)
    for _ in range(conv_num-1):
        filters *= 2
        x = convolutional_block(x=x, filters=filters)

    for _ in range(res_num):
        x = residual_block(x=x, filters=filters, kernel_size=3)
  
    for _ in range(upsample_num):
        filters = max(64, int(filters/2))
        x = upsample_block(x, filters)
    initializer = tf.random_normal_initializer(0., 0.02)
    # Mapping features to RGB
    outputs = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer, activation='tanh')(x)
    model = models.Model(inputs, outputs)
    return model