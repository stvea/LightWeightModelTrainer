class NetConfig:
    IMAGE_SIZE = 384
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    GPU = "6"
    LABEL_SMOOTHING = 0.1
    BATCH_SIZE = 64
    EPOCHS_TRAIN = 5
    EPOCHS_FINE = 300
    FINE_TUNE_START = 50
    NORMALIZATION = "per"