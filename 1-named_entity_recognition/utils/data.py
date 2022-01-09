#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 13:43
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : data.py
# @Software: PyCharm


from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="../datas"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            line = line.strip()
            if len(line)>0:
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

if __name__ == '__main__':
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    print('训练数据条数:')
    print(len(train_word_lists))
    print(len(train_tag_lists))
    print('验证集条数:')
    print(len(dev_word_lists))
    print(len(dev_tag_lists))
    print('测试数据条数:')
    print(len(test_word_lists))
    print(len(test_tag_lists))
    print(word2id)
    print(tag2id)
