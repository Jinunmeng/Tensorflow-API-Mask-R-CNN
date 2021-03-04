# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:54:51 2021

@author: mengjinjun
"""

import os
import random

trainval_percent = 0.8
train_percent = 0.75
xmlfilepath = './datasets/Data/image'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('./datasets/Data/index/trainval.txt', 'w')
ftest = open('./datasets/Data/index/test.txt', 'w')
ftrain = open('./datasets/Data/index/train.txt', 'w')
fval = open('./datasets/Data/index/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()