#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 13:41
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import os
import sys
from utils.data import  build_corpus
from utils.tools import  load_model ,extend_maps
from utils.metrics import  Metrics
from utils.tools import  extend_maps ,prepocess_data_for_lstmcrf
from train import ensemble_model
path = os.path.abspath('.')
ckpts_path = os.path.join(path ,'ckpts')
data_path = os.path.join(path , 'datas')
pkl_dirs = os.listdir(ckpts_path)
for pkl_path in pkl_dirs:
    aim_path = os.path.join(ckpts_path ,pkl_path)
    if pkl_path.find('hmm')>=0:
        HMM_MODEL_PATH = aim_path
    if pkl_path.find('crf')>=0:
        CRF_MODEL_PATH =aim_path
    if pkl_path.find('bilstm')>=0 and pkl_path.find('crf')<0:
        BiLSTM_MODEL_PATH =aim_path
    if pkl_path.find('bilstm_crf.')>=0:
        BiLSTMCRF_MODEL_PATH =aim_path

REMOVE_O = False  # 在评估的时候是否去除O标记

def main():
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train",data_dir=data_path)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False,data_dir=data_path)
    print("加载并评估hmm模型...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(test_word_lists,
                              word2id,
                              tag2id)
    metrics = Metrics(test_tag_lists, hmm_pred, remove_O=REMOVE_O)
    metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    metrics.report_confusion_matrix()  # 打印混淆矩阵
    # 加载并评估CRF模型
    print("加载并评估crf模型...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    # bilstm模型
    print("加载并评估bilstm模型...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                   bilstm_word2id, bilstm_tag2id)
    metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    ensemble_model(
        [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
        test_tag_lists
    )

if __name__ == '__main__':
    main()