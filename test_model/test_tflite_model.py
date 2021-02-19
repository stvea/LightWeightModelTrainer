from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shutil import copyfile

import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf
from data_reader import load_image, img_preprocess
from utills.base import get_FileCreateTime


def test_tflite_model(model_file, image_size=224,save=False, show_wrong=False):
    val_path = '/data2/competition/classification/val_20'

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
    top1_correct = 0
    top3_correct = 0

    for dirpath, dirnames, filenames in os.walk(val_path):
        for dirname in dirnames:
            for path, _, imgnames in os.walk(os.path.join(dirpath, dirname)):
                for imgname in imgnames:
                    label = int(dirname.split('_')[0]) - 1
                    image = load_image(os.path.join(path, imgname))
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

                    total += 1
                    if prediction_top_3[0] == label:
                        top1_correct += 1
                    if label in prediction_top_3:
                        top3_correct += 1
                    if show_wrong:
                        if prediction_top_3[0] != label:
                            print("True", label, "Pred", prediction_top_3,
                                  "Image Name:", imgname, "Top1 acc:", top1_correct / total)
                            img = Image.open(os.path.join(path, imgname))
                            plt.figure(imgname)
                            plt.imshow(img)
                            plt.axis('on')
                            plt.xlabel(imgname)

                            plt.title("True"+str(label+1)+CLASSES[label]+" Pred"+str(prediction_top_3[0]+1)+CLASSES[prediction_top_3[0]])  # 图像题目
                            plt.show()
                    else:
                        print("True", label, "Pred", prediction_top_3, "Top1 acc:", top1_correct / total)

                    if save:
                        copyfile(os.path.join(path, imgname), "/home/share/data/extra_data/" + CLASSES[label])
    print("Top1 Acc:",top1_correct/total,"Top3 Acc:",top3_correct/total)

def test():
    '''Evaluate TFLite'''
    evaluate_tflite = False
    if evaluate_tflite:
        save_wrong = True

        CLASSES = ['Portrait', 'Group Portrait', 'Kids / Infants', 'Dog', 'Cat', 'Macro / Close-up', 'Food / Gourmet',
                   'Beach', 'Mountains', 'Waterfall', 'Snow', 'Landscape', 'Underwater', 'Architecture',
                   'Sunrise / Sunset',
                   'Blue Sky', 'Overcast / Cloudy Sky', 'Greenery / Green Plants / Grass', 'Autumn Plants', 'Flower',
                   'Night Shot', 'Stage / Concert', 'Fireworks', 'Candle light', 'Neon Lights / Neon Signs', 'Indoor',
                   'Backlight / Contre-jour', 'Text / Document', 'QR Code', 'Monitor Screen']

        for i in range(len(CLASSES)):
            CLASSES[i] = "".join("".join(CLASSES[i].split()).split('/'))

        model_file = "save/" + train_name + "/" + train_name + ".tflite"
        print("Model Name:", model_file, "Create Time:", get_FileCreateTime(model_file))

        interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        floating_model = False

        if input_details[0]['dtype'] == type(np.float32(1.0)):
            floating_model = True

        # Get the size of the input / output tensors

        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        # Process test_model images and display the results

        total = 0
        top1_correct = 0
        top3_correct = 0
        top1_error = 0
        top3_error = 0

        correct = {}
        total_dict = {}
        predict_dict = {}

        top1_wrong_label = {}
        top3_wrong_label = {}
        if save_wrong:
            if not os.path.exists("save/" + train_name + "/wrong/"):
                mkdir("save/" + train_name + "/wrong")
        for dirpath, dirnames, filenames in os.walk(val_path):
            for dirname in dirnames:
                for path, _, imgnames in os.walk(os.path.join(dirpath, dirname)):
                    for imgname in imgnames:
                        label = int(dirname.split('_')[0]) - 1
                        total += 1
                        image = load_image(os.path.join(path, imgname))
                        image = img_preprocess(image, IMAGE_SIZE, NORMALIZATION, False)
                        image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
                        input_data = image

                        if floating_model:
                            input_data = np.float32(input_data)
                        else:
                            input_data = np.uint8(input_data)

                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()

                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        results = np.squeeze(output_data)

                        prediction = np.argmax(results)
                        prediction_top_3 = results.argsort()[-3:][::-1]
                        if CLASSES[label] not in total_dict:
                            total_dict[CLASSES[label]] = 1
                            correct[CLASSES[label]] = 0

                        else:
                            total_dict[CLASSES[label]] += 1

                        if CLASSES[prediction_top_3[0]] not in predict_dict:
                            predict_dict[CLASSES[prediction_top_3[0]]] = 1
                        else:
                            predict_dict[CLASSES[prediction_top_3[0]]] += 1

                        if prediction_top_3[0] == label:
                            top1_correct += 1
                            correct[CLASSES[label]] += 1
                        else:
                            if save_wrong:
                                if not os.path.exists("save/" + train_name + "/wrong/" + CLASSES[label]):
                                    mkdir("save/" + train_name + "/wrong/" + CLASSES[label])
                                copyfile(os.path.join(path, imgname),
                                         os.path.join("save/" + train_name + "/wrong/" + CLASSES[label],
                                                      CLASSES[prediction_top_3[0]] + "_" +
                                                      CLASSES[prediction_top_3[1]] + "_" +
                                                      CLASSES[prediction_top_3[2]] + "_" + ".jpg"))

                            if dirname in top1_wrong_label:
                                top1_wrong_label[dirname] += 1
                            else:
                                top1_wrong_label[dirname] = 1
                            top1_error += 1
                        if label in prediction_top_3:
                            top3_correct += 1
                        else:
                            top3_error += 1
                            if dirname in top3_wrong_label:
                                top3_wrong_label[dirname] += 1
                            else:
                                top3_wrong_label[dirname] = 1

                        print(os.path.join(path, imgname), "True", label, "Pred", prediction_top_3)
        accuracy = {}
        recall = {}
        for k in correct.keys():
            accuracy[k] = correct[k] / total_dict[k]
            recall[k] = correct[k] / (predict_dict[k])
        with open("save/" + train_name + "/tflite_test_log.txt", 'w') as f:
            f.write("Top 1 Acc " + str(top1_correct / total))
            f.write('\n')
            f.write("Top 1 Wrong " + str(sorted(top1_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))
            f.write('\n')
            f.write("Top 3 Acc " + str(top3_correct / total))
            f.write('\n')
            f.write("Top 3 Wrong " + str(sorted(top3_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)))
            f.write('\n')
            f.write("Accuracy in category:" + str(accuracy))
            f.write('\n')
            f.write("Recall in category:" + str(recall))
            f.write('\n')
        print("Top 1 Acc ", top1_correct / total)
        print("Top 1 Wrong ", sorted(top1_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        print("Top 3 Acc ", top3_correct / total)
        print("Top 3 Wrong ", sorted(top3_wrong_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        print("Accuracy in category:", accuracy)
        print("Recall in category:", recall)
