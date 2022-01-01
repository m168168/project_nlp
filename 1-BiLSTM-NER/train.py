#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/1 16:48
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import time
import torch.optim as optim
from util.build_corpus import  build_corpus
from util.utils import extend_maps,save_model
import torch
from copy import deepcopy
from  src.model import BiLSTM
from  src.util import  sort_by_lengths,tensorized,cal_loss
from  config.lstmconfig import LSTMConfig ,TrainingConfig
from  src.evaluatiing import Metrics

import os
project_path = os.path.abspath(os.path.dirname(__file__))
print(project_path)
def train():
    """
        训练模型  BI-LSTM模型
    """
    # 1：数据的读取
    print("读取数据...")
    data_path = r'./datas'
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 2：数据格式处理
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)

    # 3：模型的创建
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    config = LSTMConfig()
    model = BiLSTM(vocab_size, config.emb_size, config.hidden_size, out_size)
    print('model create ')
    # 超参数的设置
    epochs = TrainingConfig.epochs
    print_steps = TrainingConfig.print_step
    learing_rate = TrainingConfig.learing_rate
    batch_size = TrainingConfig.batche_size
    _best_val_loss = 1e18

    device = torch.device("cpu")
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)

    # 对数据集按照长度进行排序
    word_lists, tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)
    print('开始训练！')
    for e in range(1, epochs+1):
        step = 0
        losses = 0.0
        for i in range(0,len(word_lists),batch_size):
            batch_sentences = word_lists[i :i+batch_size]
            batch_tags = tag_lists[i:i+batch_size]
            model.train()
            step +=1
            # 准备数据
            tensorized_sents, lengths = tensorized(batch_sentences, word2id)
            tensorized_sents = tensorized_sents.to(device)
            targets, lengths = tensorized(batch_tags, tag2id)
            targets = targets.to(device)

            # forward
            scores =model(tensorized_sents, lengths)

            # 计算损失 更新参数
            optimizer.zero_grad()

            loss = cal_loss(scores, targets, tag2id).to(device)
            loss.backward()
            optimizer.step()
            losses +=loss.item()
            if step % TrainingConfig.print_step == 0:
                total_step = (len(word_lists) // batch_size + 1)
                print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(e, step, total_step,
                    100.*step / total_step,
                    losses /print_steps))
                losses = 0.
        # 每轮结束测试在验证集上的性能，保存最好的一个
        model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for i in range(0,len(dev_word_lists),batch_size):
                val_step +=1
                batch_sents = dev_word_lists[i:i+ batch_size]
                batch_tags = dev_tag_lists[i:i + batch_size]
                tensorized_sents, lengths = tensorized( batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(device)
                # forward
                scores = model(tensorized_sents, lengths)
                # 计算损失
                loss = cal_loss(scores, targets, tag2id).to(device)
                val_losses += loss.item()

            val_loss = val_losses / val_step
            if val_loss < _best_val_loss:
                print("保存模型...")
                best_model = deepcopy(model)
                _best_val_loss = val_loss

            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))
    model_name = 'bilstm'
    save_model(model, "./ckpts/" + model_name + ".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))

    pred_tag_lists, test_tag_lists = test(test_word_lists, test_tag_lists, word2id, tag2id,device,best_model)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=False)
    metrics.report_scores()
    metrics.report_confusion_matrix()


def test( word_lists, tag_lists, word2id, tag2id, device,best_model ):
    """返回最佳模型在测试集上的预测结果"""
    # 准备数据
    word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
    tensorized_sents, lengths = tensorized(word_lists, word2id)
    tensorized_sents = tensorized_sents.to(device)

    best_model.eval()
    with torch.no_grad():
        batch_tagids = best_model.test(
            tensorized_sents, lengths, tag2id)

    # 将id转化为标注
    pred_tag_lists = []
    id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
    for i, ids in enumerate(batch_tagids):
        tag_list = []
        for j in range(lengths[i]):
            tag_list.append(id2tag[ids[j].item()])
        pred_tag_lists.append(tag_list)
    # indices存有根据长度排序后的索引映射的信息
    # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
    # 索引为2的元素映射到新的索引是1...
    # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
    ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*ind_maps))
    pred_tag_lists = [pred_tag_lists[i] for i in indices]
    tag_lists = [tag_lists[i] for i in indices]
    return pred_tag_lists, tag_lists
def train_step(batch_sentences , batch_tags ,word2id ,tag2id,model):
    model.train()




def main():

    train()






    #4：模型的训练


    #5：模型参数的保存


if __name__ == '__main__':
    main()