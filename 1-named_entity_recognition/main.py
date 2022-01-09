#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 13:42
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import os
import sys
from utils.data import  build_corpus
from train import hmm_train,crf_train,bilstm_train ,ensemble_model
from utils.tools import  extend_maps ,prepocess_data_for_lstmcrf
path  = sys.argv[0]  # 当前执行文件路径
path = os.path.abspath('.')#获得当前工作目录
print(path)
def main():
    """训练模型，评估结果"""
    # 1读取数据
    print("读取数据...")
    data_path = os.path.join(path ,'datas')
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train",data_dir=data_path)
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False,data_dir=data_path)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False,data_dir=data_path)
    # 2训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    hmm_pred = hmm_train(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )
    # 3训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )

    # 4训练评估BI-LSTM和 BiLSTM+CRF模型
    print("正在训练评估双向LSTM模型...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    try:
        lstm_pred = bilstm_train(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            bilstm_word2id, bilstm_tag2id,
            crf=False
        )
    except  Exception as e :
        print('出现error{}'.format(e))
    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理

    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    try:
        lstmcrf_pred = bilstm_train(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            crf_word2id, crf_tag2id,
            crf=True
        )
    except  Exception as e :
        print('出现error{}'.format(e))

    # 5 上面5个模型的集成
    ensemble_model([hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],test_tag_lists)

if __name__ == '__main__':
    main()