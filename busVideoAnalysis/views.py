# -*- coding:utf-8 -*-
"""
@Author:ddz
@Date:2023/3/24 13:51
@Project:views
"""
import logging
from django.core.files import File

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
from .models import Image
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators import csrf
from django.views.decorators.csrf import csrf_exempt
import os
from datetime import datetime
from .models import Image

from segmentBaseLpsnet.predict import getMask
from detectBaseCenternet.predict import predBaseCenterNet

def py2html(request):
    context = {"p2html": ["python 2 html", "python 2 js"]}
    return render(request, "index.html", context)


# 接收POST请求数据
def postexample(request):
    ctx ={}
    if request.POST:
        ctx['rlt'] = request.POST['q']
    return render(request, "postExample.html", ctx)


@csrf_exempt
def segmentImg(request):
    img = list(Image.objects.all())[-1]
    path = getMask(f".{img.img.url}")
    #保存预测图片到数据库中
    # os.path.split(): 返回文件的路径和文件名
    dirname, filename = os.path.split(path)
    pre_img = Image(name=filename, time=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))
    with open(path,"rb") as f:
        myfile = File(f)
        pre_img.img.save(name=filename,content=myfile,save=True)
    pre_img.save()
    logging.info(pre_img.img.url)
   
    res = {"imgs": img, "label": pre_img}

    return render(request, 'imgupload.html', res)


@csrf_exempt
def uploadImg(request):
    if request.method == 'POST':
        new_img = Image(
            img=request.FILES.get('img'),
            name=request.FILES.get('img').name,
            time=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        )
        new_img.save()
        return segmentImg(request)
    return render(request, 'imgupload.html')


@csrf_exempt
def prepareImg(request):
    if request.method == 'POST':
        new_img = Image(
            img=request.FILES.get('img'),
            name=request.FILES.get('img').name,
            time=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        )
        new_img.save()
        return detectImg(request)
    return render(request, 'imgupload.html')


@csrf_exempt
def detectImg(request):
    img = list(Image.objects.all())[-1]
    path = predBaseCenterNet(f".{img.img.url}")
    # 保存预测图片到数据库中
    # os.path.split(): 返回文件的路径和文件名
    dirname, filename = os.path.split(path)
    pre_img = Image(name=filename, time=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))
    with open(path, "rb") as f:
        myfile = File(f)
        pre_img.img.save(name=filename, content=myfile, save=True)
    pre_img.save()
    logging.info(pre_img.img.url)

    res = {"imgs": img, "label": pre_img}

    return render(request, 'imgupload.html', res)