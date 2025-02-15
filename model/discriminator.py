from tensorflow.keras import layers, models

def discriminator_block(x, filters, kernel_size=4, strides=2, padding='same', norm=True, act=True):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    if norm: x = layers.BatchNormalization()(x)
    if act: x = layers.LeakyReLU(negative_slope=0.2)(x)
    return x

def build_discriminator(input_shape):
    inp = layers.Input(shape=input_shape, name='input_image')
    tar = layers.Input(shape=input_shape, name='target_image')

    x = layers.Concatenate()([inp, tar])
    
    filters = 64
    n_down = 3
    x = discriminator_block(x, filters, norm=False)
    for i in range(n_down):
        x = discriminator_block(x, filters*2**(i+1), strides=1 if i == (n_down-1) else 2)
    out = discriminator_block(x, 1, strides=1, norm=False, act=False)
    model = models.Model(inputs=[inp, tar], outputs=[out])
    return model