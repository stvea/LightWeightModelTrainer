import os
import numpy as np
import tensorflow as tf
from shutil import copyfile

from data_reader import load_image, img_preprocess
from utills.base import make_train_dataset_dir, get_FileCreateTime
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

dataset_save_path = "/home/share/data/extra_data/"
val_path = '/home/share/data/imagenet_subset/test/'
model_file = "/home/gechao/code/shufflenetv2-tf/save/t2021_02_09_15_58_mb2_224_t5_f130_nper_b128_fs25_ls/t2021_02_09_15_58_mb2_224_t5_f130_nper_b128_fs25_ls.tflite"
CLASSES = make_train_dataset_dir(dataset_save_path)

print("Model Name:", model_file, "Create Time:", get_FileCreateTime(model_file))

interpreter = tf.lite.Interpreter(model_path=str(model_file))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

total = 0
top1_correct = 0
top3_correct = 0

for dirpath, dirnames, filenames in os.walk(val_path):
    for imgname in filenames:
        image = load_image(os.path.join(dirpath, imgname))
        image = img_preprocess(image, 224, "per", False)
        test_image = image

        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        results = np.squeeze(output)

        prediction_top_3 = results.argsort()[-3:][::-1]

        total += 1
        if results[prediction_top_3[0]]>=0.50:
            print(imgname,"|Prediction:",CLASSES[prediction_top_3[0]],"|Confidence:",results[prediction_top_3[0]])
            copyfile(os.path.join(dirpath, imgname), "/home/share/data/extra_data/" + CLASSES[prediction_top_3[0]]+"/"+imgname)
