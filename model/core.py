import tensorflow as tf


def build_model(model_name, img_shape, strategy):
    with strategy.scope():
        if model_name == "mb2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=img_shape,
                include_top=False,
                alpha=1.4,
                weights='imagenet')
            base_model.trainable = False
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(192, 3, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(30, activation='softmax')
            ])
        elif model_name == "mb3":
            base_model = tf.keras.applications.MobileNetV3Large(
                input_shape=img_shape,
                include_top=False,
                alpha=1.4,
                weights='imagenet')
            base_model.trainable = False
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(192, 3, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(30, activation='softmax')
            ])
        elif model_name == "res150v2":
            base_model = tf.keras.applications.ResNet152V2(
                include_top=False,
                input_shape=img_shape,
                weights="pretrain/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
            )
            base_model.trainable = False
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(192, 3, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(30, activation='softmax')
            ])
    return model
