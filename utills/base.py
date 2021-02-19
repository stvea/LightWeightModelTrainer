import logging
import os
from shutil import copyfile

import tensorflow as tf
import matplotlib.pyplot as plt
import time


def init_train(train_name, save_path, checkpoint_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print(train_name)
    mkdir(save_path)
    mkdir(checkpoint_path)
    copyfile("net_config.py", os.path.join(save_path, "net_config.py"))

    logger.info("Start train " + train_name)
    return logger


def make_train_dataset_dir(path):
    CLASSES = ['Portrait', 'Group Portrait', 'Kids / Infants', 'Dog', 'Cat', 'Macro / Close-up', 'Food / Gourmet',
               'Beach', 'Mountains', 'Waterfall', 'Snow', 'Landscape', 'Underwater', 'Architecture', 'Sunrise / Sunset',
               'Blue Sky', 'Overcast / Cloudy Sky', 'Greenery / Green Plants / Grass', 'Autumn Plants', 'Flower',
               'Night Shot', 'Stage / Concert', 'Fireworks', 'Candle light', 'Neon Lights / Neon Signs', 'Indoor',
               'Backlight / Contre-jour', 'Text / Document', 'QR Code', 'Monitor Screen']

    for i in range(len(CLASSES)):
        CLASSES[i] = "".join("".join(CLASSES[i].split()).split('/'))
        if not os.path.exists(path + CLASSES[i]):
            os.mkdir(path + CLASSES[i])
    return CLASSES


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def timestamp_to_time(timestamp):
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d_%H:%M:%S', timeStruct)


def get_FileCreateTime(filePath):
    t = os.path.getctime(filePath)
    return timestamp_to_time(t)


def resume_from_checkpoint(net):
    net.init_model()
    latest = tf.train.latest_checkpoint("checkpoints/" + net.name + "/")
    print(latest)
    net.model.load_weights(latest)
    return net.model


def save_model(model_name, model):
    saved_model_dir = 'saved_model/' + model_name

    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)

    tf.saved_model.save(model, saved_model_dir)


# move to utils.draw_train_plot
def draw_train_history(history, name=""):
    if name != "":
        name += ":"

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title(name + 'Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title(name + 'Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
