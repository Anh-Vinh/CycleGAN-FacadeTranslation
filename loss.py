import tensorflow as tf

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.reduce_mean(tf.square(tf.ones_like(disc_real_output) - disc_real_output))
    
    generated_loss = tf.reduce_mean(tf.square(tf.zeros_like(disc_generated_output) - disc_generated_output))
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss