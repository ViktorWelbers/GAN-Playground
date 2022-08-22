import time

import numpy as np
import tensorflow as tf
from keras.models import Sequential
import configuration as cfg
from image_tools import save_images


class LossFn:
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    @classmethod
    def discriminator_loss(cls, real_output, fake_output):
        real_loss = cls.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cls.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @classmethod
    def generator_loss(cls, fake_output):
        return cls.cross_entropy(tf.ones_like(fake_output), fake_output)


class Trainer:
    generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

    def __init__(self, generator: Sequential, discriminator: Sequential):
        self.generator = generator
        self.discriminator = discriminator

    @tf.function(experimental_follow_type_hints=True)
    def step(self, images: tf.Tensor) -> tuple:
        seed = tf.random.normal([cfg.BATCH_SIZE, cfg.SEED_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(seed, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = LossFn.generator_loss(fake_output)
            disc_loss = LossFn.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset: tf.data.Dataset, epochs: int) -> None:
        fixed_seed = np.random.normal(0, 1, (cfg.PREVIEW_ROWS * cfg.PREVIEW_COLS, cfg.SEED_SIZE))
        start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for image_batch in dataset:
                t = self.step(image_batch)
                gen_loss_list.append(t[0])
                disc_loss_list.append(t[1])

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            print(f'Epoch {epoch + 1}, gen loss={g_loss},disc loss={d_loss} {time.time() - epoch_start}')
            save_images(epoch, fixed_seed, self.generator)
            self.generator.save_weights(f'{cfg.SAVED_WEIGHTS_PATH}/generator_checkpoint_{cfg.IMAGE_DIM}px')
            self.discriminator.save_weights(f'{cfg.SAVED_WEIGHTS_PATH}/discriminator_checkpoint_{cfg.IMAGE_DIM}px')
        self.discriminator.save(f'{cfg.SAVED_MODEL_PATH}/discriminator_{cfg.IMAGE_DIM}px')
        self.generator.save(f'{cfg.SAVED_MODEL_PATH}/generator_{cfg.IMAGE_DIM}px')
        print(f'Training time: {time.time() - start}')
