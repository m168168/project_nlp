#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/1 17:43
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : lstmconfig.py
# @Software: PyCharm

class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
class TrainingConfig(object):
    batche_size = 128
    learing_rate = 0.001
    epochs = 2
    print_step = 10