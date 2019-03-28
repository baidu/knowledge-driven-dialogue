Knowledge-driven Dialogue
=============================
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/baidu/knowledge-driven-dialogue/blob/master/retrieval_pt/LICENSE.md)

This is a pytorch implementation of retrieval-based model for knowledge-driven dialogue

## Requirements

* cuda=9.0
* cudnn=7.0
* python>=3.6
* numpy

## Quickstart

### Step 1: Preprocess the data

Put the data provided by the organizer under the data folder and rename them  train/dev/test.txt: 

```
./data/resource/train.txt
./data/resource/dev.txt
./data/resource/test.txt
```

### Step 2: Train the model

Train model with the following commands. 

```bash
sh run_train.sh model_name
```

3 models were supported:

- match: match, input is history and response
- match_kn: match_kn, input is history, response, chat_path, knowledge
- match_kn_gene: match_kn, input is history, response, chat_path, knowledge and generalizes target_a/target_b of goal for all inputs, replaces them with slot mark

### Step 3: Test the Model

Test model with the following commands.

```bash
sh run_test.sh model_name
```

## Note !!!

* The script run_train.sh/run_test.sh shows all the processes including data processing and model training/testing. Be sure to read it carefully and follow it.
* Building candidates by ./tools/construct_candidate.py is one of the most important step in run_train.sh/run_test.sh. You have to reimplement the core function yourself: get_candidate_for_conversation
* The files in ./data and ./model is just empty file to show the structure of the document.