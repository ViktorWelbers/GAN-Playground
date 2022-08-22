import configuration as cfg
import numpy as np
import os
from PIL import Image
import tensorflow as tf


def save_images(cnt, noise, generator: tf.keras.Model):
    image_array = np.full((
        cfg.PREVIEW_MARGIN + (cfg.PREVIEW_ROWS * (cfg.IMAGE_DIM + cfg.PREVIEW_MARGIN)),
        cfg.PREVIEW_MARGIN + (cfg.PREVIEW_COLS * (cfg.IMAGE_DIM + cfg.PREVIEW_MARGIN)), cfg.IMAGE_CHANNELS),
        255, dtype=np.uint8)

    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(cfg.PREVIEW_ROWS):
        for col in range(cfg.PREVIEW_COLS):
            r = row * (cfg.IMAGE_DIM + 16) + cfg.PREVIEW_MARGIN
            c = col * (cfg.IMAGE_DIM + 16) + cfg.PREVIEW_MARGIN
            image_array[r: r + cfg.IMAGE_DIM, c: c + cfg.IMAGE_DIM] = generated_images[image_count] * 255
            image_count += 1

    output_path = os.path.join(f'image_output/{cfg.IMAGE_DIM}px_{cfg.TRAIN_DATA_SIZE}_datapoints')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
