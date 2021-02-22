import argparse
import os
from pathlib import Path

from test_model.core import test_tflite, test_tflite_output
from utills.args import get_test_args
from utills.converter import convert_from_save_model

'''Set Start'''

test_type = "normal"
train_name = "t2021_02_22_14_39_mb2_s_224"
checkpoint_name = "checkpoint"

rebuild_tflite = False
tflite_name = "fint_model.tflite"
# tflite_name = "purning_model.tflite"
tflite_quantization_type = "full_int"

os.environ['CUDA_VISIBLE_DEVICES'] = "5"

'''Set End'''

image_size = 384
root_path = "save/" + train_name + "/"
tflite_path = root_path + tflite_name

# import config if exists
if Path(root_path + "net_config.py").exists():
    NetConfig = None
    exec('from save.' + train_name + '.net_config import *')
    image_size = NetConfig.IMAGE_SIZE

# Convert tf model to tf lite
if not Path(tflite_path).exists() or rebuild_tflite:
    tflite_path = convert_from_save_model(root_path + checkpoint_name, _image_size=image_size,
                                          type=tflite_quantization_type,
                                          tflite_save_path=tflite_path)

if test_type == "normal":
    test_tflite(tflite_path, image_size, show_wrong=True)
elif test_type == "output":
    test_tflite_output(tflite_path, image_size)
