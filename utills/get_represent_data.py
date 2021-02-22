import os
from shutil import copyfile

from PIL import Image
from data_reader import img_preprocess, load_image

data_path = "/data2/competition/classification/represent_data/"

for dirpath, dirnames, filenames in os.walk(data_path):
    for dirname in dirnames:
        for path, _, imgnames in os.walk(os.path.join(dirpath, dirname)):
            img_count = 0
            for imgname in imgnames:
                img_count += 1
                if img_count ==10:
                    break
                print(data_path + dirname.split('_')[0]+"_"+imgname)
                copyfile(os.path.join(path, imgname), data_path + dirname.split('_')[0]+"_"+imgname)
