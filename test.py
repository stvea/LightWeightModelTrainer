import argparse
import os
from pathlib import Path

from test_model.core import test_tflite, test_tflite_output
from utills.args import get_test_args
from utills.converter import convert_from_save_model

'''Set Start'''
test_type = "normal"
model_name = "model.tflite"
train_name = "t2021_02_20_11_15_mb2_384"
rebuild_tflite = True
tflite_quantization_type = "full_int"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
'''Set End'''

image_size = 384
root_path = "save/" + train_name + "/"
tflite_path = root_path + "model.tflite"

# Convert tf model to tf lite
if not Path(tflite_path).exists() or rebuild_tflite:
    tflite_path = convert_from_save_model(root_path + "checkpoint",
                                          type=tflite_quantization_type,
                                          tflite_save_path=root_path + "int_model.tflite")

# import config if exists
if Path(root_path + "net_config.py").exists():
    NetConfig = None
    exec('from save.' + train_name + '.net_config import *')
    image_size = NetConfig.IMAGE_SIZE


if test_type == "normal":
    test_tflite(tflite_path, image_size, show_wrong=True)
elif test_type == "output":
    test_tflite_output(tflite_path, image_size)
