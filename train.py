import tensorflow as tf
import time

from tensorflow.keras.callbacks import ModelCheckpoint
from IPython import display
from utils import generate_images

LAMDA = 10

@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape(persistent=True) as tape:
        # GAN Loss of Generator G mapping from X -> Y
        fake_Y = generator_G(input_image, training=True)
        
        disc_real_Y = discriminator_Y([input_image, target_image], training=True)
        disc_fake_Y = discriminator_Y([input_image, fake_Y], training=True)
        
        G_gan_loss = tf.reduce_mean(tf.square(tf.ones_like(disc_fake_Y) - disc_fake_Y))
        Y_disc_loss = discriminator_loss(disc_real_Y, disc_fake_Y)

        # GAN Loss of Generator F mapping from Y -> X
        fake_X = generator_F(target_image, training=True)
        
        disc_real_X = discriminator_X([target_image, input_image], training=True)
        disc_fake_X = discriminator_X([target_image, fake_X], training=True)

        F_gan_loss = tf.reduce_mean(tf.square(tf.ones_like(disc_fake_X) - disc_fake_X))
        X_disc_loss = discriminator_loss(disc_real_X, disc_fake_X)

        # Cycle-Consistent Loss
        cycled_Y = generator_G(fake_X, training=True)
        cycled_X = generator_F(fake_Y, training=True)
        G_l1_loss = tf.reduce_mean(tf.abs(target_image - cycled_Y))
        F_l1_loss = tf.reduce_mean(tf.abs(input_image - cycled_X))

        cycle_loss = G_l1_loss + F_l1_loss
        total_loss = G_gan_loss + F_gan_loss + cycle_loss * LAMDA
    
    # Calculate gradients for generators
    G_gradients = tape.gradient(total_loss, generator_G.trainable_variables)
    F_gradients = tape.gradient(total_loss, generator_F.trainable_variables)

    # Calculate gradients for discriminators
    Y_gradients = tape.gradient(Y_disc_loss, discriminator_Y.trainable_variables)
    X_gradients = tape.gradient(X_disc_loss, discriminator_X.trainable_variables)

    # Apply gradients to the optimizers
    optimizer_G.apply_gradients(zip(G_gradients, generator_G.trainable_variables))
    optimizer_F.apply_gradients(zip(F_gradients, generator_F.trainable_variables))

    optimizer_Y.apply_gradients(zip(Y_gradients, discriminator_Y.trainable_variables))
    optimizer_X.apply_gradients(zip(X_gradients, discriminator_X.trainable_variables))
    
    return G_gan_loss, Y_disc_loss, F_gan_loss, X_disc_loss, total_loss

def fit(train_ds, test_ds, start_epoch=0, end_epoch=100):
    example_input, example_target = next(iter(test_ds.take(1)))

    for epoch in range(start_epoch, end_epoch):
        train_G_loss = 0.0
        train_Y_loss = 0.0
        train_F_loss = 0.0
        train_X_loss = 0.0
        total_loss = 0.0
        
        # Training and display the progress with every 10%
        print(f"Epoch {epoch + 1}/{end_epoch}")
        start = time.time()

        for step, (input_image, target) in train_ds.enumerate():
            
            G_loss, Y_loss, F_loss, X_loss, total = train_step(input_image, target)
            train_G_loss += G_loss
            train_Y_loss += Y_loss
            train_F_loss += F_loss
            train_X_loss += X_loss
            total_loss += total
  
            if (step+1) % (num_sample // 10) == 0:
                print('.', end='', flush=True)
             
        display.clear_output(wait=True)
        
        print(f"Time: {time.time() - start:.2f} sec\n"
        f"G Loss: {train_G_loss/num_sample}, Y Loss: {train_Y_loss/num_sample}\n"
        f"F Loss: {train_F_loss/num_sample}, X Loss: {train_X_loss/num_sample}\n"
        f"Total Loss: {total_loss/num_sample}")

        # Display an instance
        generate_images(generator_G, example_input, example_target)
        generate_images(generator_F, example_target, example_input)
        
        # Save the checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))