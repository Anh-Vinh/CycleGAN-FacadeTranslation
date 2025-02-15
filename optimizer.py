import tensorflow as tf

class LinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, step_per_epoch):
        self.lr = lr
        self.final_lr = 0
        self.step_per_epoch = step_per_epoch

    def __call__(self, step):
        epoch = tf.cast(step, tf.float32) // tf.cast(self.step_per_epoch, tf.float32)

        # Decay from the 100th epoch
        return tf.cond(
            epoch < 100,
            lambda: self.lr,
            lambda: self.lr - ((self.lr - self.final_lr) * (epoch - 100) / epoch)
        )