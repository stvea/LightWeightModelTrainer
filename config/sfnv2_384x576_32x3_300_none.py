import tensorflow as tf
from model.shufflenet import ShufflenetV2


class MyNet:
    name = "sfnv2_384x576_32x3_300_pretrain-none"
    height = 384
    width = 576
    epochs = 300

    def __init__(self,batch_size=32):
        self.batch_size = batch_size

    def init_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model = ShufflenetV2(30, True)
        model.build(input_shape=(self.batch_size, self.width, self.height, 3))
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                ['accuracy'],
                [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="Top1_accuracy")],
                [tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="Top3_accuracy")]
            ])
        self.model = model
