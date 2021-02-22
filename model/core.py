import tensorflow as tf
import tensorflow_model_optimization as tfmot


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
                tf.keras.layers.Dropout(0.7),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(30, activation='softmax'),

            ])
            print(base_model.summary())
            exit()
        elif model_name == "mb2_s":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=img_shape,
                include_top=False,
                alpha=1.4,
                weights='imagenet')
            base_model.trainable = False

            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(192, 3, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(30, activation='softmax'),

            ])
            new_model = tf.keras.Model(model.input, model.output)
            model = new_model
        elif model_name == "mb2_quant":

            base_model = tf.keras.applications.MobileNetV2(
                input_shape=img_shape,
                include_top=False,
                alpha=1.4,
                weights='imagenet')
            # base_model.trainable = False

            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(192, 3, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(30, activation='softmax'),

            ])
            quant_model = tf.keras.Sequential()
            for layer in base_model.layers:
                print(layer.name)
                if layer.name == "block_2_add":
                    quant_model.add(layer)
                else:
                    quant_model.add(
                        tfmot.quantization.keras.quantize_annotate_layer(layer)
                    )
            # quant_model.add(
            #     tfmot.quantization.keras.quantize_annotate_layer(base_model)
            # )
            # quant_model.add(tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(192, 3, activation='relu')))
            # quant_model.add(tf.keras.layers.Dropout(0.5))
            # quant_model.add(tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.GlobalAveragePooling2D()))
            # quant_model.add(tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(30, activation="softmax")))
            print(quant_model.summary())
            model = quant_model
            # q_feature_extractor = tfmot.quantization.keras.quantize_model(base_model)
            # q_feature_extractor.summary()
            #
            #
            # model_top = tf.keras.models.Sequential([
            #     tf.keras.layers.InputLayer(input_shape=(7,7,1792)),
            #     tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(192, 3, activation='relu')),
            #     tf.keras.layers.Dropout(0.5),
            #     tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.GlobalAveragePooling2D()),
            #     tfmot.quantization.keras.quantize_annotate_layer(
            #         tf.keras.layers.Dense(30, activation="softmax")
            #     )]
            # )
            #
            # q_model_top = tfmot.quantization.keras.quantize_apply(model_top)
            # q_model_top.summary()
            #
            # inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
            # x = q_feature_extractor(inputs)
            # outputs = q_model_top(x)
            #
            # model = tf.keras.Model(inputs=inputs, outputs=outputs)
            #
            # print(model.summary())
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
