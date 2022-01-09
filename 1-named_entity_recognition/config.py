#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 14:05
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : config.py
# @Software: PyCharm
# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 16
    # 学习速率
    lr = 0.001
    epoches = 1
    print_step = 5
class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
class OtherConfig(object):
    remove_O = False # 是否移除O

