#!/usr/bin/python3
#--coding:utf-8--
#@file       : Process.py
#@author     : chenwanyuan
#@date       :2019/06/19/
#@description:
#@other      :
import json
from bert import tokenization
import tensorflow as tf

class ClassifierProcess():
    def __init__(self,frozen_graph_filename,bert_vocab_file,schema_file,sequence_length,gpu_memory_fraction=0.2):

        self.id2schema = {}
        with open(schema_file, encoding='utf8') as fp:
            for i, line in enumerate(fp):
                data = json.loads(line.strip())
                s_t = data['subject_type']
                o_t = data['object_type']
                p = data['predicate']
                spo_name = '{}-{}-{}'.format(s_t, p, o_t)
                self.id2schema[i] = spo_name
        def load_graph(frozen_graph_filename):
            # We parse the graph_def file
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # We load the graph_def in the default graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def)
            return graph

        def load_sess(graph):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
            return tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))

        self.bert_tokenizer = tokenization.FullTokenizer(
            vocab_file=bert_vocab_file, do_lower_case=True)
        self.sequence_length = sequence_length
        self.graph = load_graph(frozen_graph_filename)
        self.sess = load_sess(self.graph)
        self.logits_tensor = self.graph.get_operation_by_name("import/logits").outputs[0]
        self.input_ids_tensor = self.graph.get_operation_by_name("import/input_ids").outputs[0]
        self.input_mask_tensor = self.graph.get_operation_by_name("import/input_mask").outputs[0]
        self.segment_ids_tensor = self.graph.get_operation_by_name("import/segment_ids").outputs[0]

    def process(self,content):

        tokens = self.bert_tokenizer.tokenize(content)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = len(input_ids) * [1]

        if len(input_ids) < self.sequence_length:
            input_ids = input_ids + [0] * (self.sequence_length - len(input_ids))
            input_mask = input_mask + [0] * (self.sequence_length - len(input_mask))
        else:
            input_ids = input_ids[:self.sequence_length]
            input_mask = input_mask[:self.sequence_length]
        segment_ids = [0] * self.sequence_length

        input_ids = [input_ids]
        input_mask = [input_mask]
        segment_ids = [segment_ids]


        preds = self.sess.run(self.logits_tensor, feed_dict={
            self.input_ids_tensor: input_ids,
            self.input_mask_tensor:input_mask,
            self.segment_ids_tensor:segment_ids
        })
        preds = preds[0]
        schema2score = {}
        for i, pred in enumerate(preds):
            schema2score[self.id2schema[i]] = float('{:.4f}'.format(pred))
        return schema2score


class ExtractorProcess():
    def __init__(self,frozen_graph_filename,bert_vocab_file,sequence_length,
        gpu_memory_fraction=0.2):

        self.label2id = {'0': 0, 'B-S': 1, "I-S": 2, 'B-O': 3, 'I-O': 4}
        self.id2label = {}
        for label, id in self.label2id.items():
            self.id2label[id] = label
        def load_graph(frozen_graph_filename):
            # We parse the graph_def file
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            # We load the graph_def in the default graph
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def)
            return graph

        def load_sess(graph):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
            return tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))

        self.bert_tokenizer = tokenization.FullTokenizer(
            vocab_file=bert_vocab_file, do_lower_case=True)
        self.sequence_length = sequence_length
        self.graph = load_graph(frozen_graph_filename)
        self.sess = load_sess(self.graph)
        self.decode_tags_tensor = self.graph.get_operation_by_name("import/decode_tags").outputs[0]
        self.input_ids_tensor = self.graph.get_operation_by_name("import/input_ids").outputs[0]
        self.input_mask_tensor = self.graph.get_operation_by_name("import/input_mask").outputs[0]
        self.segment_ids_tensor = self.graph.get_operation_by_name("import/segment_ids").outputs[0]
        self.sequence_length_tensor = self.graph.get_operation_by_name("import/sequence_length").outputs[0]
    def process(self,spo_name,content):

        tokens = self.bert_tokenizer.tokenize(content)
        schema_name_tokens = self.bert_tokenizer.tokenize(spo_name)
        tokens_not_unk = self.bert_tokenizer.tokenize_not_UNK(content)

        input_tokens = ['[CLS]'] + schema_name_tokens + ['[SEP]'] + tokens + ['[SEP]']
        input_tokens_not_unk = ['[CLS]'] + schema_name_tokens + ['[SEP]'] + tokens_not_unk + ['[SEP]']

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * (len(schema_name_tokens) + 2) + (len(tokens) + 1) * [1]
        input_mask = len(input_ids) * [1]
        length = min(self.sequence_length,len(input_ids))

        if len(input_ids) < self.sequence_length:
            input_ids = input_ids + [0] * (self.sequence_length - len(input_ids))
            input_mask = input_mask + [0] * (self.sequence_length - len(input_mask))
            segment_ids = segment_ids + [0] * (self.sequence_length - len(segment_ids))
        else:
            input_ids = input_ids[:self.sequence_length]
            input_mask = input_mask[:self.sequence_length]
            segment_ids = segment_ids[:self.sequence_length]
        input_ids = [input_ids]
        input_mask = [input_mask]
        segment_ids = [segment_ids]

        preds = self.sess.run(self.decode_tags_tensor, feed_dict={
            self.input_ids_tensor: input_ids,
            self.input_mask_tensor: input_mask,
            self.segment_ids_tensor: segment_ids,
            self.sequence_length_tensor:[length]
        })

        preds = preds[0]
        ners = []
        last_label = ''
        b_pos = -1
        e_pos = -1
        idx = 0
        for i in range(len(schema_name_tokens) + 2,length,1):

            id = preds[i]
            label = self.id2label.get(id)
            #print(label,input_tokens_not_unk[i])
            if label == '0':
                if last_label:
                    ners.append(
                        (last_label, ''.join(input_tokens_not_unk[b_pos: e_pos+1]).replace('##', ''),idx))

                last_label = ''
                b_pos = -1
                e_pos = -1
                continue

            pre,label = label.split('-')
            if pre == 'B':
                idx = i-len(schema_name_tokens) -2
                if last_label and label != last_label:
                    ners.append((last_label,
                                 ''.join(input_tokens_not_unk[b_pos:e_pos + 1]).replace('##', ''),idx))
                last_label = label
                b_pos = i
                e_pos = i

            if  pre == 'I':
                if label == last_label:
                    e_pos += 1
                else:
                    if last_label:
                        ners.append((last_label,
                                     ''.join(input_tokens_not_unk[b_pos:e_pos+1]).replace('##', ''),idx))
                    last_label = ''
                    b_pos = -1
                    e_pos = -1

        if last_label:
            ners.append((last_label,''.join(input_tokens_not_unk[b_pos:e_pos+1]).replace('##', ''),idx))

        return ners

class TextProcess():
    def __init__(self,param):
        self.param = param
    def init(self):
        self.classifier_process = ClassifierProcess(**self.param['classifier'])
        self.extractor_process = ExtractorProcess(**self.param['extractor'])
    def process(self,content):
        schema2score = self.classifier_process.process(content)
        results = []
        for schema,score in schema2score.items():
            if True:# score>0.3:
                ners = self.extractor_process.process(schema,content)

                subject_set = set()
                object_set = set()
                for ner in ners:
                    label ,entity,position = ner
                    if len(entity)>1:
                        if label == 'S':
                            subject_set.add(entity)
                        if label == 'O':
                            object_set.add(entity)
                s_t,p,o_t = schema.split('-')
                for s in subject_set:
                    for o in object_set:
                        results.append({
                            'subject_type':s_t,
                            'object_type':o_t,
                            'predicate':p,
                            'subject':s,
                            'object':o,
                            'score':score
                        })

        return results


if __name__ == '__main__':
    import os, re
    import json
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    spo_name = '歌曲-作曲-人物'
    content = '丝角蝗科，Oedipodidae，昆虫纲直翅目蝗总科的一个科'
    # classifier = ClassifierProcess('./data/graph/classifier.pb',bert_vocab_file='./data/chinese_L-12_H-768_A-12/vocab.txt',schema_file='./data/all_50_schemas',sequence_length=256)
    # print(classifier.process(content))
    #extractor = ExtractorProcess('./data/graph/extractor.pb',bert_vocab_file='./data/chinese_L-12_H-768_A-12/vocab.txt',sequence_length=256)
    #print(extractor.process(spo_name,content))

    param = {
        'classifier':{
            'frozen_graph_filename':'./data/graph/classifier.pb',
            'bert_vocab_file': './data/chinese_L-12_H-768_A-12/vocab.txt',
            'schema_file':'./data/all_50_schemas',
            'sequence_length':256
        },
        'extractor': {
            'frozen_graph_filename': './data/graph/extractor.pb',
            'bert_vocab_file': './data/chinese_L-12_H-768_A-12/vocab.txt',
            'sequence_length': 256
        }
    }
    process = TextProcess(param)
    process.init()
    print(process.process(content))

