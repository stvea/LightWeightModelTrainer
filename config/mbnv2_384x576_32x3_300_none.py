import tensorflow as tf


class MyNet:
    name = "mbnv2_384x576_32x3_300_none"
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
        model = tf.keras.applications.MobileNetV2(
            input_shape=(384, 576, 3),
            weights=None,
            classes=30,
            classifier_activation='softmax'
        )
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=metrics)
        self.model = model
