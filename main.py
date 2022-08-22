import os

import tensorflow as tf
from sklearn.utils import resample
import configuration as cfg
from dataset_gen import generate_training_data_from_images
from models import build_generator, build_discriminator
from train import Trainer


def main():
    training_data_path = os.path.join(cfg.TRAIN_DATA_PATH, f'training_data_{cfg.IMAGE_DIM}_{cfg.IMAGE_DIM}.npy')
    training_data = generate_training_data_from_images(training_data_path)

    training_data = resample(training_data, n_samples=cfg.TRAIN_DATA_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(cfg.BUFFER_SIZE).batch(cfg.BATCH_SIZE)

    if os.path.isdir(os.path.join(cfg.SAVED_MODEL_PATH, f'generator_{cfg.IMAGE_DIM}px')):
        print('Loading models')
        generator = tf.keras.models.load_model(os.path.join(cfg.SAVED_MODEL_PATH, f'generator_{cfg.IMAGE_DIM}px'))
        generator.summary()
        discriminator = tf.keras.models.load_model(os.path.join(cfg.SAVED_MODEL_PATH, f'discriminator_{cfg.IMAGE_DIM}px'))
        discriminator.summary()
    else:
        print('Building models')
        generator = build_generator(cfg.SEED_SIZE, cfg.IMAGE_CHANNELS)
        image_shape = (cfg.IMAGE_DIM, cfg.IMAGE_DIM, cfg.IMAGE_CHANNELS)
        discriminator = build_discriminator(image_shape)

    trainer = Trainer(generator, discriminator)
    trainer.train(dataset, cfg.EPOCHS)


if __name__ == '__main__':
    main()
