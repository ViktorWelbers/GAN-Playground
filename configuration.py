import os

GENERATE_RES = 4
IMAGE_DIM = 32 * GENERATE_RES
IMAGE_CHANNELS = 3

PREVIEW_ROWS = 5
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

SEED_SIZE = 100


RAW_DATA_PATH = os.path.join('raw_data', 'image')
TRAIN_DATA_PATH = os.path.join('datasets')
SAVED_MODEL_PATH = './saved_models'
SAVED_WEIGHTS_PATH = './weights'
TRAIN_DATA_SIZE = 25000
EPOCHS = 15
BATCH_SIZE = 32
BUFFER_SIZE = 60000
