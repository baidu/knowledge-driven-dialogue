#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: conversation_strategy.py
"""

import sys

sys.path.append("../")
import interact
from convert_conversation_corpus_to_model_text import preprocessing_for_one_conversation


def load():
    """
    load
    """
    return interact.load_model()


def predict(model, text):
    """
    predict
    """
    model_text, candidates = \
        preprocessing_for_one_conversation(text.strip(),
                                           candidate_num=50,
                                           use_knowledge=True,
                                           topic_generalization=True,
                                           for_predict=True)

    for i, text_ in enumerate(model_text):
        score = interact.predict(model, text_)
        candidates[i] = [candidates[i], score]

    candidate_legal = sorted(candidates, key=lambda item: item[1], reverse=True)
    return candidate_legal[0][0]


def main():
    """
    main
    """
    model = load()
    for line in sys.stdin:
        response = predict(model, line.strip())
        print(response)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
