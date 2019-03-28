#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: construct_candidate.py
"""

import sys
import json
import collections


def get_candidate_for_conversation(conversation, candidate_num=10):
    """
    get candidate for conversation
    !!! you have to reimplement the function yourself !!!
    """
    return ["这 是 一个 示例 候选，你 必须 实现 自己 的 候选 构造 ！"] * candidate_num


def construct_candidate_for_corpus(corpus_file, candidate_file, candidate_num=10):
    """
    construct candidate for corpus

    case of data in corpus_file:
    {
        "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
        "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
        "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
                    "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"]
    }

    case of data in candidate_file:
    {
        "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
        "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
        "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
                    "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"],
        "candidate": ["我 说 的 是 休 · 劳瑞 。",
                      "我 说 的 是 休 · 劳瑞 。"]
    }
    """
    fout_text = open(candidate_file, 'w')
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            conversation = json.loads(line.strip(), encoding="utf-8", \
                                 object_pairs_hook=collections.OrderedDict)
            candidates = get_candidate_for_conversation(conversation,
                                                        candidate_num=candidate_num)
            conversation["candidate"] = candidates

            conversation = json.dumps(conversation, ensure_ascii=False)
            fout_text.write(conversation + "\n")

    fout_text.close()


def main():
    """
    main
    """
    construct_candidate_for_corpus(sys.argv[1], sys.argv[2], int(sys.argv[3]))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
