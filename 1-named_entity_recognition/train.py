#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 14:00
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import os
import time
import datetime
from collections import Counter
from utils.tools import *
from utils.metrics import Metrics
from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model

path = os.path.abspath('.')
ckpts_path = os.path.join(path ,'ckpts')
pkl_dirs = os.listdir(ckpts_path)
HMM_MODEL_PATH = r''
CRF_MODEL_PATH = r''
BiLSTM_MODEL_PATH = r''
BiLSTMCRF_MODEL_PATH = r''
for pkl_path in pkl_dirs:
    aim_path = os.path.join(ckpts_path ,pkl_path)
    if pkl_path.find('hmm')>=0:
        HMM_MODEL_PATH = aim_path
    if pkl_path.find('crf')>=0:
        CRF_MODEL_PATH =aim_path
    if pkl_path.find('bilstm.')>=0 and pkl_path.find('crf')<0:
        BiLSTM_MODEL_PATH =aim_path
    if pkl_path.find('bilstm_crf.')>=0:
        BiLSTMCRF_MODEL_PATH =aim_path

def hmm_train(train_data, test_data, word2id, tag2id, remove_O=False):
    """
    训练hmm模型
    :param train_data:
    :param test_data:
    :param word2id:
    :param tag2id:
    :param remove_O:
    :return:
    """
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data
    # 创建模型
    print('创建模型HMM...')
    hmm_model = HMM(len(tag2id), len(word2id))
    # 模型的训练
    print('训练模型HMM')
    hmm_model.train(train_word_lists,train_tag_lists,word2id, tag2id)
    # 模型参数的保存
    print('保存模型HMM')
    save_model(hmm_model, "./ckpts/hmm.pkl")
    # 评估hmm模型
    print('评估模型HMM')
    pred_tag_lists = hmm_model.test(test_word_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists

def crf_train(train_data, test_data, remove_O=False):
    """
     训练CRF模型
    :param rain_data:
    :param test_data:
    :param remove_O:
    :return:
    """
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data
    # 创建模型
    print('创建模型CRF...')
    crf_model = CRFModel()
    # 模型的训练
    print('训练模型CRF')
    crf_model.train(train_word_lists, train_tag_lists)
    # 模型参数的保存
    print('保存模型CRF')
    save_model(crf_model, "./ckpts/crf.pkl")
    # 评估hmm模型
    print('评估模型CRF')
    pred_tag_lists = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    return pred_tag_lists

def bilstm_train(train_data, dev_data, test_data,word2id, tag2id, crf=True, remove_O=False):
    """
    训练bilstm 模型
    :param train_data:
    :param dev_data:
    :param test_data:
    :param word2id:
    :param tag2id:
    :param crf:
    :param remove_O:
    :return:
    """
    # 加载数据
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    # 创建模型 ,如果参数存在，直接加载模型继续训练
    if crf is True:
        if os.path.exists(BiLSTMCRF_MODEL_PATH) :
            print('加载训练好的参数')
            bilstm_model = load_model(BiLSTM_MODEL_PATH)
        else :
            print('创建新的模型')
            bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)

    else:
        if os.path.exists(BiLSTM_MODEL_PATH):
            print('加载训练好的参数')
            bilstm_model = load_model(BiLSTM_MODEL_PATH)
        else :
            print('创建新的模型')
            bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    # 模型的训练
    print('训练模型...')
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)
    model_name = "bilstm_crf" if crf else "bilstm"
    # 模型参数的保存
    print('保存模型')
    today = datetime.date.today().strftime('%y%m%d')
    # save_model(bilstm_model, "./ckpts/" + model_name +"_"+today+ ".pkl")
    save_model(bilstm_model, "./ckpts/" + model_name + ".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    return pred_tag_lists

def ensemble_model(results, targets, remove_O=False ):
    """
    多种模型的集成，投票
    :param results:
    :param targets:
    :param remove_O:
    :return:
    """
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])
    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)
    print("Ensemble 四个模型的结果如下：")
    metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()



