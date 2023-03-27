# -*- coding:utf-8 -*-
"""
@Author:ddz
@Date:2023/3/21 16:16
@Project:predict.py
"""
import logging

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import namedtuple
from collections import OrderedDict
from uuid import uuid1

from segmentBaseLpsnet.lpsnet import get_lspnet_s
from segmentBaseLpsnet.lpsnet import upsample


def getMask(fimage):

    Cls = namedtuple('cls', ['name', 'id', 'color'])
    Clss = [
        Cls('road', 0, (128, 64, 128)),
        Cls('sidewalk', 1, (244, 35, 232)),
        Cls('building', 2, (70, 70, 70)),
        Cls('wall', 3, (102, 102, 156)),
        Cls('fence', 4, (190, 153, 153)),
        Cls('pole', 5, (153, 153, 153)),
        Cls('traffic light', 6, (250, 170, 30)),
        Cls('traffic sign', 7, (220, 220, 0)),
        Cls('vegetation', 8, (107, 142, 35)),
        Cls('terrain', 9, (152, 251, 152)),
        Cls('sky', 10, (70, 130, 180)),
        Cls('person', 11, (220, 20, 60)),
        Cls('rider', 12, (255, 0, 0)),
        Cls('car', 13, (0, 0, 142)),
        Cls('truck', 14, (0, 0, 70)),
        Cls('bus', 15, (0, 60, 100)),
        Cls('train', 16, (0, 80, 100)),
        Cls('motorcycle', 17, (0, 0, 230)),
        Cls('bicycle', 18, (119, 11, 32))
    ]
    # 定义颜色列表（RGB值）
    color_dict = {}
    for cls in Clss:
        color_dict[cls.id] = cls.color

    # 创建一个空白的调色板（256个RGB值）
    palette = [0] * 256 * 3

    # 将颜色列表填充到调色板中（从索引1开始，索引为零表示背景）
    for i in color_dict.keys():
        palette[(i + 1) * 3:(i + 1) * 3 + 3] = color_dict.get(i)

    # 将调色板转换为字节类型
    palette_bytes = bytes(palette)

    img, msk = imgPredict(fimage)
    img_ = img.permute(0, 2, 3, 1).cpu().numpy() * 255
    msk_ = msk.cpu().numpy()

    for img, msk in zip(img_, msk_):
        irgb = Image.fromarray(img.astype(np.uint8))
        imsk = Image.fromarray(msk.astype(np.uint8))

        # 将mask转换为调色板模式，并使用自定义的调色板替换默认的黑白调色板
        imsk.putpalette(palette_bytes)

        # 将两张图片合并，使用alpha通道控制透明度（可根据需要调整）
        result = Image.blend(irgb.convert('RGBA'), imsk.convert('RGBA'), alpha=0.318)

        fname = f"temp/img/seg_{str(uuid1())[:8]}.png"
    
        result.save(fname)
        return fname


def imgPredict(fimage="segmentBaseLpsnet/img/test2.jpg"):
    # show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_lspnet_s(num_classes=19)
    state_dict = torch.load("segmentBaseLpsnet/checkpoint/lpsnet_distribute_latest.pth", map_location=device)

    model = model.to(device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        # load params
    model.load_state_dict(new_state_dict)

    model.eval()

    images = np.array(Image.open(fimage))/255.0
    images = torch.FloatTensor(images).permute(2, 0, 1).unsqueeze(dim=0)

    with torch.no_grad():

        images = images.to(device)

        preds = model(images)
        preds = upsample(preds, images.shape[-2:])
    return images, preds.argmax(dim=1)


if __name__ == '__main__':
    pass
    # show()



