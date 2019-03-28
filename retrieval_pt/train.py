#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: train.py
"""

import time
import numpy as np
import torch
import source.inputters.data_provider as reader
from source.models.retrieval_model import RetrievalModel
from args import base_parser
from args import print_arguments
from source.utils.optimizers import Optimizer


def build_data(args):
    """
    build train and dev data
    """
    task_name = args.task_name.lower()
    processor = reader.MatchProcessor(data_dir=args.data_dir,
                                      task_name=task_name,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case)

    train_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='train',
        epoch=args.epoch,
        shuffle=True,
        device=args.gpu)
    num_train_examples = processor.get_num_examples(phase='train')

    dev_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='dev',
        epoch=1,
        shuffle=False,
        device=args.gpu)
    num_dev_examples = processor.get_num_examples(phase='dev')

    max_train_steps = args.epoch * num_train_examples // args.batch_size
    warmup_steps = int(max_train_steps * args.warmup_proportion)

    train_data = [train_data_generator, num_train_examples]
    dev_data = [dev_data_generator, num_dev_examples]

    return processor, [train_data, dev_data], warmup_steps


def build_optim(model, args, warmup_steps=4000, checkpoint=None):
    """
    build optimizer
    """
    if checkpoint is not None:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        optim = Optimizer(
            args.optim, args.learning_rate, args.max_grad_norm,
            lr_decay=args.learning_rate_decay,
            start_decay_steps=args.start_decay_steps,
            decay_steps=args.decay_steps,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            adagrad_accum=args.adagrad_accumulator_init,
            decay_method=args.decay_method,
            warmup_steps=warmup_steps,
            model_size=args.emb_size)

    optim.set_parameters(model)

    return optim


def build_model(args, num_labels, checkpoint=None):
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

    if checkpoint is not None:
        retrieval_model.load_state_dict(checkpoint['model'])

    return retrieval_model


def drop_checkpoint(model, optimizer, args, process, stats):
    """
    called conditionally each epoch to save a snapshot.
    """
    checkpoint = {
        'model': model.state_dict(),
        'args': args,
        'optim': optimizer
    }
    batch, epoch, example = process
    loss, acc = stats

    torch.save(checkpoint,
               '%s/model_loss_%.4f_acc_%.4f_e%d_i%d_b%d.pt'
               % (args.checkpoints, loss, acc, epoch, example, batch))


def train_model(model, optimizer, train_data, dev_data, processor, args):
    """
    called conditionally each epoch to save a snapshot.
    """
    train_data_generator, num_train_examples = train_data
    dev_data_generator, num_dev_examples = dev_data

    model.train()
    time_begin = time.time()
    total_cost, total_acc, total_num = [], [], []
    for batch_id, data in enumerate(train_data_generator()):
        inputs, _ = data["inputs"]
        positions, _ = data["positions"]
        senttypes, _ = data["senttypes"]
        labels, _ = data["labels"]
        knowledge, _ = data["knowledge"] if args.use_knowledge and "knowledge" in data else [None, None]

        model.zero_grad()
        avg_loss, avg_acc, cur_num = model(inputs, positions, senttypes, labels, knowledge)
        avg_loss.backward()
        optimizer.step()

        total_cost.append(float(avg_loss) * cur_num)
        total_acc.append(float(avg_acc) * cur_num)
        total_num.append(cur_num)
        if batch_id % args.skip_steps == 0:
            time_end = time.time()
            used_time = time_end - time_begin
            current_example, current_epoch = processor.get_train_progress()
            print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                  "ave acc: %f, speed: %f steps/s" %
                  (current_epoch, current_example, num_train_examples,
                   batch_id, np.sum(total_cost) / np.sum(total_num),
                   np.sum(total_acc) / np.sum(total_num),
                   args.skip_steps / used_time))
            time_begin = time.time()
            total_cost, total_acc, total_num = [], [], []

        if batch_id % args.validation_steps == 0:
            model.eval()
            total_dev_cost, total_dev_acc, total_dev_num = [], [], []
            for dev_id, dev_data in enumerate(dev_data_generator()):
                inputs, _ = dev_data["inputs"]
                positions, _ = dev_data["positions"]
                senttypes, _  = dev_data["senttypes"]
                labels, _  = dev_data["labels"]
                knowledge, _  = dev_data["knowledge"] \
                    if args.use_knowledge and "knowledge" in dev_data else [None, None]

                avg_loss, avg_acc, cur_num = model(inputs, positions, senttypes, labels, knowledge)

                total_dev_cost.append(float(avg_loss) * cur_num)
                total_dev_acc.append(float(avg_acc) * cur_num)
                total_dev_num.append(cur_num)

            dev_loss = np.sum(total_dev_cost) / np.sum(total_dev_num)
            dev_acc = np.sum(total_dev_acc) / np.sum(total_dev_num)
            print("valid eval: ave loss: %f, ave acc: %f" % (dev_loss, dev_acc))

            drop_checkpoint(model, optimizer, args,
                            [batch_id, current_epoch, current_example],
                            [dev_loss, dev_acc])

            model.train()


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

    processor, [train_data, dev_data], warmup_steps = build_data(args)

    args.voc_size = len(open(args.vocab_path, 'r').readlines())
    num_labels = len(processor.get_labels())

    checkpoint = None
    if args.init_checkpoint:
        print('Loading checkpoint from %s' % args.init_checkpoint)
        checkpoint = torch.load(args.init_checkpoint,
                                map_location=lambda storage, loc: storage)

    retrieval_model = build_model(args, num_labels, checkpoint)

    optimizer = build_optim(retrieval_model, args, warmup_steps, checkpoint)

    train_model(retrieval_model, optimizer, train_data, dev_data, processor, args)


if __name__ == '__main__':
    args = base_parser()
    print_arguments(args)
    main(args)
