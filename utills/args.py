import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description="Test model and tflite")
    parser.add_argument('-m', '--model-name', default='mb2', type=str)
    return parser.parse_args()

def get_train_checkpoint_args():
    parser = argparse.ArgumentParser(description="Test model and tflite")
    parser.add_argument('-tn', '--train-name', default='mb2', type=str)
    return parser.parse_args()

def get_test_args():
    parser = argparse.ArgumentParser(description="Test model and tflite")
    parser.add_argument('-m', '--model-path', default='/data2/competition/classification/train_2_18', type=str)
    parser.add_argument('-s', '--image-size', default=224, type=int)
    parser.add_argument('-mt', '--model-type', default='tflite', type=str)
    parser.add_argument('-tt', '--test-type', default='normal', type=str)
    return parser.parse_args()