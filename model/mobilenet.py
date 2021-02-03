import os
import tensorflow as tf

from data_reader import *

os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"
strategy = tf.distribute.MirroredStrategy()

train_ds = get_train_data()
val_ds = get_val_data()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.Adam()

with strategy.scope():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
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
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            ['accuracy'],
            [tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="Top3 accuracy")]
        ]
    )

checkpoint_dir = './training_checkpoints_pretrain_imagenet'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

history = model.fit(train_ds,
                    epochs=10,
                    shuffle=False,
                    validation_data=val_ds,
                    validation_steps=len(val_ds),
                    callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                               tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)],
                    verbose=1)
