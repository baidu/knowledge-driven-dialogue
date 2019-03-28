#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: args.py
"""

import six
import argparse


def base_parser():
    """
    define argument parser add common arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for running classifier.")
    #
    parser.add_argument('--voc_size', type=int, default=13309,
        help="""Total token number in batch for training.""")
    parser.add_argument('--emb_size', type=int, default=256,
        help=""". (Word embedding size.""")
    parser.add_argument('--layers', type=int, default=4,
        help="""Number of layers in encoder.""")
    parser.add_argument('--rnn_size', type=int, default=256,
        help="""Size of rnn hidden states.""")
    parser.add_argument('--heads', type=int, default=8,
        help="""Number of heads in TransformerEncoder.""")
    parser.add_argument('--bridge', action="store_true",
        help="""additional layer between the last encoder state and the first decoder state""")
    parser.add_argument('--rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU'],
        help="""The gate type to use in the RNNs""")
    parser.add_argument('--max_seq_len', type=int, default=256,
        help="""Number of word of the longest sequence.""")
    parser.add_argument('--share_embedding', type=bool, default=False, choices=[True, False],
        help="""Share the word embeddings between context encoder and knowledge encoder""")
    parser.add_argument('--epoch', type=int, default=30,
        help="""Number of epoches for training.""")
    parser.add_argument('--task_name', type=str, default='match_kn_gene',
        help="""Task name for training. Options [match|match_kn|match_kn_gene].""")
    parser.add_argument('--use_knowledge', action="store_true",
        help="""Whether to use knowledge or not.""")
    parser.add_argument('--batch_size', type=int, default=128,
        help="""Maximum batch size for training/testing.""")
    parser.add_argument('--init_checkpoint', type=str, default=None,
        help="""init checkpoint to resume training from or to test the model.""")
    parser.add_argument('--output', type=str, default="./output/predict.txt",
        help="""File to output the predict scores.""")
    parser.add_argument('--weight_decay', type=float, default=0.01,
        help="""Weight decay rate for L2 regularizer.""")
    parser.add_argument('--checkpoints', type=str, default=None,
        help="""Path to save checkpoints.""")
    parser.add_argument('--vocab_path', type=str, default="./dict/gene.dict",
        help="""Vocabulary path. """)
    parser.add_argument('--data_dir', type=str, default="./data",
        help="""Path of train/dev/test data. """)
    parser.add_argument('--skip_steps', type=int, default=1000,
        help="""The steps interval to print loss.""")
    parser.add_argument('--save_steps', type=int, default=7000,
        help="""The steps interval to save checkpoints.""")
    parser.add_argument('--validation_steps', type=int, default=7000,
        help="""The steps interval to evaluate model performance on validation set.""")
    parser.add_argument('--use_gpu', action='store_true',
        help="""If set, use GPU for training.""")
    parser.add_argument('--gpu', type=int, default=0,
        help="""gpu id""")
    parser.add_argument('--do_lower_case', type=bool, default=True, choices=[True, False],
        help="""Whether to lower case the input text. Should be True for uncased 
        models and False for cased models.""")
    parser.add_argument('--optim', default='adam', choices=['sgd', 'adagrad', 'adadelta', 'adam'],
        help="""Optimization method.""")
    parser.add_argument('--adagrad_accumulator_init', type=float, default=0,
        help="""Initializes the accumulator values in adagrad.
        Mirrors the initial_accumulator_value option
        in the tensorflow adagrad (use 0.1 for their default).""")
    parser.add_argument('--max_grad_norm', type=float, default=5,
        help="""If the norm of the gradient vector exceeds this,
        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('--dropout', type=float, default=0.3,
        help="Dropout probability")
    parser.add_argument('--adam_beta1', type=float, default=0.9,
        help="""The beta1 parameter used by Adam.""")
    parser.add_argument('--adam_beta2', type=float, default=0.999,
        help="""The beta2 parameter used by Adam.""")
    # learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help="""Starting learning rate. Recommended settings: sgd = 1, adagrad = 0.1,
        adadelta = 1, adam = 0.001. Learning rate used to train with warmup. """)
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
        help="""proportion warmup.""")
    parser.add_argument('--learning_rate_decay', type=float, default=0.5,
        help="""If update_learning_rate, decay learning rate by
        this much if steps have gone past start_decay_steps""")
    parser.add_argument('--start_decay_steps', type=int, default=50000,
        help="""Start decaying every decay_steps after start_decay_steps""")
    parser.add_argument('--decay_steps', type=int, default=10000,
        help="""Decay every decay_steps""")
    parser.add_argument('--decay_method', type=str, default="noam", choices=['noam', 'none'],
        help="Use a custom decay rate.")
    parser.add_argument('--warmup_steps', type=int, default=4000,
        help="""Number of warmup steps for custom decay.""")
    args = parser.parse_args()
    return args


def print_arguments(args):
    """ print arguments """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

