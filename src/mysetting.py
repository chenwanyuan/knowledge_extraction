#!/usr/bin/python3
#--coding:utf-8--
#@file       : mysetting.py
#@author     : chenwanyuan
#@date       :2019/06/19/
#@description:
#@other      :

MODEL_PARAM={
        'classifier':{
            'frozen_graph_filename':'../data/graph/classifier.pb',
            'bert_vocab_file': '../data/chinese_L-12_H-768_A-12/vocab.txt',
            'schema_file':'../data/all_50_schemas',
            'sequence_length':256
        },
        'extractor': {
            'frozen_graph_filename': '../data/graph/extractor.pb',
            'bert_vocab_file': '../data/chinese_L-12_H-768_A-12/vocab.txt',
            'sequence_length': 256
        }
    }
SERVER_PORT = 7070
SERVER_PROCESS_NUM = 1
LOG_NAME = "ke"
LOG_DIR = '../log/'
LOG_FILE = 'ke.log'