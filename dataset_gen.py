import os
import time

import numpy as np
from PIL import Image

import configuration as cfg


def get_all_images_in_folders(data_path: os.path) -> tuple[list, list]:
    data = []
    for path, folder, files in os.walk(data_path):
        if files:
            for f in files:
                data.append((os.path.join(path, f), f[:-4]))
    data = sorted(data, key=lambda x: x[1])
    img_paths, img_ids = list(zip(*data))
    return img_paths, img_ids


def generate_training_data_from_images(training_binary_path: os.path) -> np.ndarray:
    if os.path.isfile(training_binary_path):
        return np.load(training_binary_path)
    start = time.time()
    training_data = []
    img_paths, img_ids = get_all_images_in_folders(cfg.RAW_DATA_PATH)

    for path in img_paths:
        image = Image.open(path).resize((cfg.IMAGE_DIM, cfg.IMAGE_DIM), Image.ANTIALIAS)
        arr = np.asarray(image, dtype="int32")
        if arr.shape == (cfg.IMAGE_DIM, cfg.IMAGE_DIM, cfg.IMAGE_CHANNELS):
            training_data.append(arr)

    training_data = np.stack(training_data)
    training_data = training_data.astype(np.float32)
    training_data = training_data / 127.5 - 1.

    np.save(training_binary_path, training_data)
    print(f'Image preprocess time: {time.time() - start}')
    return training_data
