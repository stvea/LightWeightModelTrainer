import tensorflow as tf


class MyNet:
    name = "mbnv2_384x576_32x3_300_imgnet"
    height = 384
    width = 576
    epochs = 300

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def init_model(self):
        optimizer = tf.keras.optimizers.Adam()
        metrics =[
            ['accuracy'],
            [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="Top1_accuracy")],
            [tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="Top3_accuracy")]
        ]

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=( self.width, self.height, 3),
            alpha=1.4,
            weights='imagenet',
            include_top=False
        )
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(30, activation='softmax')
        ])
        base_model.trainable = False
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=metrics)
        self.model = model
