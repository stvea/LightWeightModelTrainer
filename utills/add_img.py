import os
import random

from PIL import Image


def addImg(raw_img, mark_img, save_name):
    print("qrcode/" + "qr_" + save_name)
    raw_img.paste(mark_img, (int(random.random() * (raw_img.size[0] - mark_img.size[0])),
                             int(random.random() * (raw_img.size[1] - mark_img.size[1]))))
    raw_img.save("qrcode/" + "qr_" + save_name)


text_img_path = "/home/gechao/code/shufflenetv2-tf/data/train_new/28_Text_Documents/"

for dirpath, dirnames, filenames in os.walk(text_img_path):
    for imgname in filenames:
        qrcode_img_path = "data/qrcode.jpg"
        try:
            addImg(Image.open(os.path.join(dirpath, imgname)), Image.open(qrcode_img_path), imgname)
        except:
            continue
