from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from test_model.test_tflite_model import test_tflite_model

train_name = "ti2021_02_19_10_01_mb2_224_t5_f300_nper_b64_fs25_ls"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
model_file = "save/test/model.tflite"
# model_file = "save/"+train_name+"/"+train_name+".tflite"
test_tflite_model(model_file,show_wrong=True,image_size=384)
