#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/1 16:50
# @Author  : xiaoxiaoshutong
# @Email   : ****
# @Site    : 
# @File    : model.py
# @Software: PyCharm

import torch
import torch.nn as nn
from  torch.nn.utils.rnn import  pad_packed_sequence ,pack_padded_sequence

class BiLSTM(nn.Module):
    def __init__(self,vocab_size ,emb_size ,hidden_size , out_size):
        """

        :param vocab_size:
        :param emb_size:
        :param hidden_size:
        :param out_size:
        """
        super(BiLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size , emb_size)
        self.bilstm = nn.LSTM(emb_size,hidden_size,batch_first=True ,bidirectional=True)
        self.Liner = nn.Linear(2*hidden_size ,out_size)
    def forward(self,sentences_tensor , lengths):
        embedding = self.embedding(sentences_tensor) # [Batch_size ，sentence_length，embedding_size]
        packed = pack_padded_sequence(embedding, lengths ,batch_first=True)
        rnn_out ,_ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.Liner(rnn_out)  # [B, L, out_size]
        return  scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids