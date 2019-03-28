#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: retrieval_model.py
"""


import torch.nn as nn
from source.utils.utils import accuracy
from source.modules.attention import Attention
from source.encoders.rnn_encoder import RNNEncoder
from source.encoders.transformer import TransformerEncoder


class RetrievalModel(nn.Module):
    """ RetrievalModel """
    def __init__(self,
                 emb_size=1024,
                 n_layer=12,
                 n_head=1,
                 voc_size=10005,
                 max_position_seq_len=1024,
                 sent_types=2,
                 num_labels=2,
                 dropout=0.3,
                 use_knowledge=False,
                 share_embedding=False,
                 padding_idx=0,
                 use_gpu=True):

        super(RetrievalModel, self).__init__()

        self.emb_size = emb_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.voc_size = voc_size
        self.max_position_seq_len = max_position_seq_len
        self.sent_types = sent_types
        self.num_labels = num_labels
        self.dropout = dropout
        self.use_knowledge = use_knowledge
        self.share_embedding = share_embedding
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu

        self.embeddings = [nn.Embedding(self.voc_size, self.emb_size, self.padding_idx),
                           nn.Embedding(self.max_position_seq_len, self.emb_size),
                           nn.Embedding(self.sent_types, self.emb_size)]

        self.transformer_encoder = TransformerEncoder(self.n_layer,
                                                      self.emb_size,
                                                      self.n_head,
                                                      self.emb_size * 4,
                                                      self.dropout,
                                                      self.embeddings)

        if self.use_knowledge:
            if self.share_embedding:
                self.knowledge_embeddings = self.embeddings[0]
            else:
                self.knowledge_embeddings = \
                    nn.Embedding(self.voc_size, self.emb_size, self.padding_idx)

            self.rnn_encoder = RNNEncoder(rnn_type="GRU",
                                          bidirectional=True,
                                          num_layers=1,
                                          input_size=self.emb_size,
                                          hidden_size=self.emb_size,
                                          dropout=self.dropout,
                                          embeddings=self.knowledge_embeddings,
                                          use_bridge=True)

            self.attention = Attention(query_size=self.emb_size,
                                       memory_size=self.emb_size,
                                       hidden_size=self.emb_size,
                                       mode="dot",
                                       project=True)

        self.middle_linear = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.Tanh()
        )

        self.final_linear = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.emb_size, self.num_labels)
        )

        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

        if self.use_gpu:
            self.cuda()

    def score(self, inputs, positions, senttypes, knowledge=None):
        """ score """
        _, outputs = self.transformer_encoder(inputs, positions, senttypes)
        outputs = self.middle_linear(outputs[:, 0, :].unsqueeze(1))
        if self.use_knowledge and knowledge is not None:
            _, memory_bank, _ = self.rnn_encoder(knowledge)
            outputs, _ = self.attention(outputs, memory_bank)

            del memory_bank

        outputs = self.final_linear(outputs.squeeze(1))

        return outputs

    def forward(self, inputs, positions, senttypes, labels, knowledge=None):
        """ forward """
        outputs = self.score(inputs, positions, senttypes, knowledge)
        outputs = self.softmax(outputs)
        labels = labels.view(-1)
        loss = self.criterion(outputs, labels)
        acc, num = accuracy(outputs, labels)

        del outputs

        return loss, acc, num
