import tensorflow as tf


class NetConfig:
    IMAGE_SIZE = 224
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    GPU = "7"
    LABEL_SMOOTHING = 0.1
    BATCH_SIZE = 64
    EPOCHS_TRAIN = 5
    EPOCHS_FINE = 300
    FINE_TUNE_START = 25
    NORMALIZATION = "per"

    @staticmethod
    def img_preprocess(image, image_size, normalization, train=True):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [image_size, image_size])

        if normalization == "255":
            image = (image / 255.0)
        elif normalization == "imagenet":
            mean_RGB = tf.constant([123.68, 116.779, 109.939], dtype=tf.float32)
            std_RGB = tf.constant([58.393, 57.12, 57.375], dtype=tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.subtract(image, mean_RGB)
            image = tf.divide(image, std_RGB)
        elif normalization == "per":
            image = tf.image.per_image_standardization(image)

        if train:
            image = tf.image.random_brightness(image, max_delta=0.3)
            image = tf.image.random_contrast(image, lower=0.8, upper=1)

        return image
