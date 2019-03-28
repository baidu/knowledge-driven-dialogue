#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: predict.py
"""


import torch
import torch.nn.functional as F
import source.inputters.data_provider as reader
from source.models.retrieval_model import RetrievalModel
from args import base_parser
from args import print_arguments


def build_data(args):
    """
    build test data
    """
    task_name = args.task_name.lower()
    processor = reader.MatchProcessor(data_dir=args.data_dir,
                                      task_name=task_name,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case)

    test_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='test',
        epoch=1,
        shuffle=False,
        device=args.gpu)
    num_test_examples = processor.get_num_examples(phase='test')

    test_data = [test_data_generator, num_test_examples]

    return processor, test_data


def build_model(args, num_labels):
    """
    build retrieval model
    """
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


def main(args):
    """
    main
    """
    args.use_gpu = torch.cuda.is_available() and args.use_gpu and args.gpu >= 0
    args.gpu = args.gpu if args.use_gpu else -1
    torch.cuda.set_device(args.gpu)

    if 'kn' in args.task_name:
        args.use_knowledge = True
    else:
        args.use_knowledge = False

    processor, test_data = build_data(args)

    args.voc_size = len(open(args.vocab_path, 'r').readlines())
    num_labels = len(processor.get_labels())

    retrieval_model = build_model(args, num_labels)

    out_scores = open(args.output, 'w')
    test_data_generator, num_test_examples = test_data
    for batch_id, data in enumerate(test_data_generator()):
        inputs, _ = data["inputs"]
        positions, _ = data["positions"]
        senttypes, _ = data["senttypes"]
        labels, _ = data["labels"]
        knowledge, _ = data["knowledge"] if args.use_knowledge and "knowledge" in data else [None, None]

        outputs = retrieval_model.score(inputs, positions, senttypes, knowledge)

        outputs = F.softmax(outputs, dim=1)

        scores = outputs.tolist()

        for i, score in enumerate(scores):
            out_scores.write("%.4f\n" % (score[1]))
            out_scores.flush()

    out_scores.close()


if __name__ == '__main__':
    args = base_parser()
    print_arguments(args)
    main(args)
