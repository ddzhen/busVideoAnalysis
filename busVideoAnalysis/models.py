# -*- coding:utf-8 -*-
"""
@Author:ddz
@Date:2023/3/24 13:49
@Project:models
"""
import logging

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


from django.db import models


class Image(models.Model):
    id = models.IntegerField(primary_key=True)
    # 图片
    img = models.ImageField(upload_to='img')
    name = models.CharField(max_length=20)
    # 创建时间
    time = models.DateTimeField(auto_now_add=True)


if __name__ == "__main__":
    pass