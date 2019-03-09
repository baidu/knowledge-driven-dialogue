#!/bin/bash
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# generalizes target_a/target_b of goal for all outputs, replaces them with slot mark
TOPIC_GENERALIZATION=1

# set python path according to your actual environment
pythonpath='python'

# the prefix of the file name used by the model, must be consistent with the configuration in network.py
prefix=demo

# put all data set that used and generated for training under this folder: datapath
# for more details, please refer to the following data processing instructions
datapath=./data

# in train stage, use "train.txt" to train model, and use "dev.txt" to eval model
# the "train.txt" and "dev.txt" are the original data provided by the organizer and
# need to be placed in this folder: datapath/resource/
# the following preprocessing will generate the actual data needed for model training
# DATA_TYPE = "train" or "dev"
datatype=(train dev)

# data preprocessing
for ((i=0; i<${#datatype[*]}; i++))
do
    # ensure that each file is in the correct path
    #     1. put the data provided by the organizers under this folder: datapath/resource/
    #            - the data provided consists of three parts: train.txt dev.txt test.txt
    #            - the train.txt and dev.txt are session data, the test.txt is sample data
    #            - in train stage, we just use the train.txt and dev.txt
    #     2. the sample data extracted from session data is in this folder: datapath/resource/
    #     3. the text file required by the model is in this folder: datapath
    #     4. the topic file used to generalize data is in this directory: datapath
    corpus_file=${datapath}/resource/${datatype[$i]}.txt
    sample_file=${datapath}/resource/sample.${datatype[$i]}.txt
    text_file=${datapath}/${prefix}.${datatype[$i]}
    topic_file=${datapath}/${prefix}.${datatype[$i]}.topic

    # step 1: firstly have to convert session data to sample data
    ${pythonpath} ./tools/convert_session_to_sample.py ${corpus_file} ${sample_file}

    # step 2: convert sample data to text data required by the model
    ${pythonpath} ./tools/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file} ${topic_file} ${TOPIC_GENERALIZATION}
done

# step 3: in train stage, we just use train.txt and dev.txt, so we copy dev.txt to test.txt for model training
cp ${datapath}/${prefix}.dev ${datapath}/${prefix}.test

# step 4: train model, you can find the model file in ./models/ after training
${pythonpath} ./network.py --gpu 0 > log.txt