#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: interact.py
"""

import torch
import torch.nn.functional as F
import source.inputters.data_provider as reader
from source.inputters.data_provider import load_dict
from source.inputters.data_provider import MatchProcessor
from source.inputters.data_provider import preprocessing_for_one_line
from source.models.retrieval_model import RetrievalModel
from args import base_parser


args = base_parser()
args.use_gpu = torch.cuda.is_available() and args.gpu >= 0
args.gpu = args.gpu if args.use_gpu else -1
torch.cuda.set_device(args.gpu)

load_dict(args.vocab_path)
args.voc_size = len(reader.VOC_DICT)
label_list = MatchProcessor.get_labels()
num_labels = len(label_list)


def load_model():
    """
    load model function
    """
    if 'kn' in args.task_name:
        args.use_knowledge = True
    else:
        args.use_knowledge = False

    retrieval_model = RetrievalModel(emb_size=args.emb_size,
                                     n_layer=args.layers,
                                     n_head=args.heads,
                                     voc_size=args.voc_size,
                                     sent_types=2,
                                     num_labels=num_labels,
                                     dropout=args.dropout,
                                     use_knowledge=args.use_knowledge,
                                     share_embedding=args.share_embedding,
                                     padding_idx=0,
                                     use_gpu=args.use_gpu)

    checkpoint = torch.load(args.init_checkpoint,
                            map_location=lambda storage, loc: storage)

    retrieval_model.load_state_dict(checkpoint['model'])

    return retrieval_model


def predict(model_handle, text):
    """
    predict score function
    """
    data = preprocessing_for_one_line(text, label_list,
                                      args.task_name, args.max_seq_len, args.gpu)

    inputs, _ = data["inputs"]
    positions, _ = data["positions"]
    senttypes, _ = data["senttypes"]
    labels, _ = data["labels"]
    knowledge, _ = data["knowledge"] if args.use_knowledge and "knowledge" in data else [None, None]

    outputs = model_handle.score(inputs, positions, senttypes, knowledge)
    outputs = F.softmax(outputs, dim=1)

    scores = outputs.tolist()

    return scores[0][1]
