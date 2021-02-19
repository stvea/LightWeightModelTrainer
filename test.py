import argparse

from test_model.core import test_tflite, test_tflite_output
from utills.args import get_test_args

args = get_test_args()

if args.test_type == "normal":
    test_tflite(args.model_path, args.image_size)
elif args.test_type == "output":
    test_tflite_output(args.model_path, args.image_size)
