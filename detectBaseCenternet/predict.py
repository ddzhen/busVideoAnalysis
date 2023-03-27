# -*- coding:utf-8 -*-
"""
@Author:ddz
@Date:2023/3/27 15:59
@Project:forecast
"""
import logging
from uuid import uuid1
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

from detectBaseCenternet.centernet import CenterNet
from PIL import Image


class padBaseImage():
    def __init__(self, fimage, pv=(255, 255, 255)):
        # fimage 图片文件
        self.img = Image.open(fimage)
        self.w = self.img.width
        self.h = self.img.height
        self.pv = pv  # 填充值

    def padding(self, pw, ph):
        img_ = Image.new('RGB', (self.w + pw, self.h + ph), self.pv)
        Image.Image.paste(img_, self.img, (pw // 2, ph // 2))

        return img_


class padSquareImage(padBaseImage):
    def __init__(self, fname, pv=(255, 255, 255)):
        super(padSquareImage, self).__init__(fname, pv)

    def squre_padding(self):
        pw = max(0, self.h - self.w)
        ph = max(0, self.w - self.h)
        img = self.padding(pw, ph)
        return img


def predBaseCenterNet(fimage="detectBaseCenternet/img/3.jpg"):

    oimg = padSquareImage(fimage)
    img = oimg.squre_padding()
    img = img.resize((512, 512))

    obj = CenterNet(cuda=False)
    res = obj.detect_image(image=img)
    fname = f"temp/img/seg_{str(uuid1())[:8]}.png"

    res.save(fname)

    # res.show()
    return fname


if __name__ == "__main__":
    predBaseCenterNet()
    pass