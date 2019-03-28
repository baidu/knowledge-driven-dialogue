#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2017-Present OpenNMT
# Copyright (c) 2019 Baidu.com, Inc.
#
################################################################################
"""
File: optimizers.py
"""

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def build_optim(model, args):
    """ Build optimizer """

    optim = Optimizer(
        args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay,
        start_decay_steps=args.start_decay_steps,
        decay_steps=args.decay_steps,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        adagrad_accum=args.adagrad_accumulator_init,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps,
        model_size=args.rnn_size)

    optim.set_parameters(model)

    return optim


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
      warmup_steps (int, option): parameter for `noam` decay
      model_size (int, option): parameter for `noam` decay
    """

    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 warmup_steps=4000,
                 model_size=None):

        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def set_parameters(self, model):
        """ ? """
        params = [p for p in model.parameters() if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.learning_rate)

        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(
                self.params,
                lr=self.learning_rate,
                initial_accumulator_value=self.adagrad_accum)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(params, lr=self.learning_rate)

        elif self.method == 'adam':
            self.optimizer = optim.Adam(params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            lr_scale = (
                self.model_size ** (-0.5) *
                min(self._step ** (-0.5),
                    self._step * self.warmup_steps ** (-1.5)))
        # Decay based on start_decay_steps every decay_steps
        elif self.start_decay_steps is not None:
            step = self._step - self.start_decay_steps
            lr_scale = (self.lr_decay ** (
                max(step + self.decay_steps, 0) // self.decay_steps))
        else:
            lr_scale = 1

        self.learning_rate = lr_scale * self.original_lr
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
            if self.max_grad_norm:
                clip_grad_norm_(group['params'], self.max_grad_norm)

        self.optimizer.step()
