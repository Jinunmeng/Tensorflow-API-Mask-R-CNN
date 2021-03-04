#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2021/3/3 19:41
@author: March
@contact: jinunmeng@163.com
@File : xml2csv.py.py
@Software: PyCharm
"""

import os
import cv2
import datetime
import json
import getArea
import glob
import numpy as np
import base64
import io
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps


from sys import argv
from base64 import b64encode
from json import dumps

def _get_point(all_points_x, all_points_y):
    point = []
    for zzz in range(len(all_points_x)):
        point.append([float(all_points_x[zzz]),float(all_points_y[zzz])])
    return point

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1], #原作者把这里写反了。造成了他原来的代码转化出来的标注文件生成的的mask维度不对。
            "height": image_size[0],#原作者把这里写反了。造成了他原来的代码转化出来的标注文件生成的的mask维度不对。
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info

def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,# float
        "bbox": bounding_box,# [x,y,width,height]
        "segmentation": segmentation# [polygon]
    }
    return annotation_info

def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]

def vgg_to_labelme(img_dir, out_dir):
    '''
    :param img_dir: 图片和json文件位于同一个目录下
    :param out_dir: 输出路径
    :return:
    '''
    # 获取所有json文件全路径
    vgg_json = glob.glob(os.path.join(img_dir, "*.json"))

    # 遍历文件夹中所有的json文件
    for num, json_file in enumerate(vgg_json):
        coco_output = {}
        coco_output['version'] = "4.2.9"
        coco_output['flags'] = {}
        coco_output['shapes'] = []
        shapes_info = []  # shapes的结果


        json_file_name = os.path.basename(json_file)
        img_file_name = json_file_name.replace(".json", ".jpg")
        img_path = os.path.join(img_dir, img_file_name)
        img = cv2.imread(img_path)
        # 用 base64 将图片保存为字符串
        # 读取二进制图片，获得原始字节码，注意 'rb'
        with open(img_path, 'rb') as jpg_file:
            byte_content = jpg_file.read()
        # 把原始字节码编码成 base64 字节码
        base64_bytes = b64encode(byte_content)
        # 将 base64 字节码解码成 utf-8 格式的字符串
        base64_string = base64_bytes.decode('utf-8')

        data = json.load(open(json_file))
        for img_id, key in enumerate(data.keys()):
            filename = data[key]['filename']
            regions = data[key]["regions"]
            for region in regions:
                shapes_sample = {}
                shape_att = region["shape_attributes"]
                if shape_att['name'] == 'polygon':
                    shapes_sample['label'] = 'building'
                    all_points_x = shape_att["all_points_x"]
                    all_points_y = shape_att["all_points_y"]
                    area = getArea.GetAreaOfPolyGon(all_points_x, all_points_y)
                    points = _get_point(all_points_x, all_points_y)
                    shapes_sample['points'] = points
                    shapes_sample['group_id'] = None
                    shapes_sample['shape_type'] = "polygon"
                    shapes_sample['flags'] = {}
                    #shapes_sample['area'] = area
                    if area < 2:
                        continue
                    shapes_info.append(shapes_sample)
        coco_output["shapes"] = shapes_info
        coco_output['imagePath'] = os.path.basename(img_file_name)
        coco_output['imageData'] = base64_string
        coco_output['imageHeight'] = img.shape[0]
        coco_output['imageWidth'] = img.shape[1]

        # 把结果导出json文件
        out_path = os.path.join(out_dir, json_file_name)
        with open(out_path, 'w', encoding='utf8') as file_obj:
            json.dump(coco_output, file_obj, indent=2)


if __name__== '__main__':

    img_path = 'C:\\Users\\Desktop\\anhui_only\\val'
    out_path = 'C:\\Users\\Desktop\\anhui_only\\val_labelme'
    vgg_to_labelme(img_path, out_path)

