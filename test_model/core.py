from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from shutil import copyfile

import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf
from data_reader import load_image, img_preprocess
from utills.base import get_FileCreateTime


def test_tflite(model_file,
                image_size=224,
                save_plabel=False,
                show_wrong=False,
                save_wrong_img=False):
    val_path = '/data2/competition/classification/val_true'

    CLASSES = ['Portrait', 'Group Portrait', 'Kids / Infants', 'Dog', 'Cat', 'Macro / Close-up', 'Food / Gourmet',
               'Beach', 'Mountains', 'Waterfall', 'Snow', 'Landscape', 'Underwater', 'Architecture', 'Sunrise / Sunset',
               'Blue Sky', 'Overcast / Cloudy Sky', 'Greenery / Green Plants / Grass', 'Autumn Plants', 'Flower',
               'Night Shot', 'Stage / Concert', 'Fireworks', 'Candle light', 'Neon Lights / Neon Signs', 'Indoor',
               'Backlight / Contre-jour', 'Text / Document', 'QR Code', 'Monitor Screen']

    for i in range(len(CLASSES)):
        CLASSES[i] = "".join("".join(CLASSES[i].split()).split('/'))

    print("Model Name:", model_file, "Create Time:", get_FileCreateTime(model_file))

    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    total = 0
    total_time = 0
    top1_correct = 0
    top3_correct = 0

    for dirpath, dirnames, filenames in os.walk(val_path):
        for dirname in dirnames:
            for path, _, imgnames in os.walk(os.path.join(dirpath, dirname)):
                for imgname in imgnames:
                    total += 1

                    label = int(dirname.split('_')[0]) - 1
                    image = load_image(os.path.join(path, imgname))
                    image = img_preprocess(image, image_size, "per", False)

                    test_image = image
                    start_time = time.time()
                    if input_details['dtype'] == np.uint8:
                        input_scale, input_zero_point = input_details["quantization"]
                        test_image = test_image / input_scale + input_zero_point

                    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
                    interpreter.set_tensor(input_details["index"], test_image)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details["index"])[0]
                    total_time += time.time()-start_time
                    print("Time:",total_time/total)
                    results = np.squeeze(output)
                    prediction_top_3 = results.argsort()[-3:][::-1]

                    if prediction_top_3[0] == label:
                        top1_correct += 1
                    if label in prediction_top_3:
                        top3_correct += 1
                    if show_wrong:
                        if prediction_top_3[0] != label:
                            print("True", label, "Pred", prediction_top_3,
                                  "Image Name:", imgname, "Top3 Acc:", results.argsort()[-3:])
                            img = Image.open(os.path.join(path, imgname))
                            plt.figure(imgname)
                            plt.imshow(img)
                            plt.axis('on')
                            plt.xlabel(imgname)

                            plt.title(
                                "True" + str(label + 1) + CLASSES[label] + " Pred" + str(prediction_top_3[0] + 1) +
                                CLASSES[prediction_top_3[0]])  # 图像题目
                            plt.show()
                    else:
                        print("True", label, "Pred", prediction_top_3, "Top3 Acc:", results.argsort()[-3:])

                    if save_plabel:
                        copyfile(os.path.join(path, imgname), "/home/share/data/extra_data/" + CLASSES[label])
                    if save_wrong_img:
                        pass
    print("Top1 Acc:", top1_correct / total, "Top3 Acc:", top3_correct / total)


def test_tflite_output(model_file,
                       image_size=224):
    val_path = 'data/val_official/'
    print("Model Name:", model_file, "Create Time:", get_FileCreateTime(model_file))

    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for dirpath, dirnames, filenames in os.walk(val_path):
        filenames.sort(key=lambda x: int(x.split('.')[0]))
        for imgname in filenames:
            image = load_image(os.path.join(dirpath, imgname))
            image = img_preprocess(image, image_size, "per", False)
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
            print(prediction_top_3[0] + 1)
