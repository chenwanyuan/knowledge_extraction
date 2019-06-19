#!/usr/bin/python3
#--coding:utf-8--
#@file       : classifier.py
#@author     : chenwanyuan
#@date       :2019/06/10/
#@description:
#@other      :

from bert import modeling
import json
import tqdm
from tensorflow.python.framework import graph_util
from tensorflow.contrib.rnn import LSTMCell
import tensorflow as tf
from bert import tokenization
from bert import optimization
from collections import defaultdict
import os
import traceback



bert_config_file = './data/chinese_L-12_H-768_A-12/bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = './data/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_tokenizer = tokenization.FullTokenizer(
        vocab_file="./data/chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=True)


schema_file = './data/all_50_schemas'
ckpt_dir = './classfier'
train_file = './data/train_classifier.tfrecord'
corpus_file = './data/train_data.json'
schema2id = {}
id2schema = {}
sequence_length = 256

with open(schema_file, encoding='utf8') as fp:
    for i,line in enumerate(fp):
        data = json.loads(line.strip())
        s_t = data['subject_type']
        o_t = data['object_type']
        p = data['predicate']
        spo_name = '{}-{}-{}'.format(s_t, p, o_t)
        schema2id[spo_name] = i
        id2schema[i] = spo_name
        print(i,spo_name)

def write_tfrecord(file,tfrecord_file,sequence_length):
    '''
    :param file: 语料文件
    :param tfrecord_file:  tfrecord文件用于测试或者训练使用
    :return:
    '''

    global bert_tokenizer

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    num = 0
    with open(file, encoding='utf-8') as fp:
        for line in fp:
            if num > 2000:break
            line = line.strip()
            data = json.loads(line)
            text = data['text']
            spo_list = data['spo_list']

            label_ids = [0] * len(schema2id)

            for spo in spo_list:
                s_t = spo['subject_type']
                o_t = spo['object_type']
                p = spo['predicate']
                spo_name = '{}-{}-{}'.format(s_t, p, o_t)
                label_ids[schema2id[spo_name]] = 1

            tokens = bert_tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = len(input_ids) * [1]

            if len(input_ids) < sequence_length:
                input_ids = input_ids + [0] * (sequence_length - len(input_ids))
                input_mask = input_mask + [0] * (sequence_length - len(input_mask))
            else:
                input_ids = input_ids[:sequence_length]
                input_mask = input_mask[:sequence_length]
            segment_ids = [0] * sequence_length
            #print(len(input_ids),len(label_ids),len(input_mask),len(segment_ids))
            features = {}
            features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
            features["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(label_ids)))
            features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_mask)))
            features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(segment_ids)))

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            num += 1
            writer.write(tf_example.SerializeToString())

    writer.close()
    print(num)

def input_fn_builder(input_file,sequence_length,num_labels,batch_size=32,is_training=True,epochs=1):

    name_to_features = {
        "input_ids": tf.FixedLenFeature([sequence_length], tf.int64),
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
        "input_mask": tf.FixedLenFeature([sequence_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([sequence_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    dataset = tf.data.TFRecordDataset(input_file)
    if is_training:
        dataset = dataset.repeat(epochs)
        dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.map(lambda record: _decode_record(record, name_to_features))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    return dataset

class BertClassifier():
    def __init__(self,bert_config,features,num_labels,init_checkpoint,is_training=False,learning_rate=None,num_train_steps=None,num_warmup_steps=None):
        self.input_ids,self.input_mask,self.segment_ids,self.label_ids = features["input_ids"],\
                                                features["input_mask"] , \
                                                features["segment_ids"],\
                                                features.get("label_ids", None)


        print(features)



        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()

        intent_hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, intent_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())


        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        if is_training:
            label_ids = tf.cast(self.label_ids, tf.float32)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=label_ids)
            self.total_loss = tf.reduce_mean(tf.reduce_sum(per_example_loss, axis=-1))
        self.logits = tf.sigmoid(logits,name='logits')



        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if is_training:
            self.train_op = optimization.create_optimizer(
                self.total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            self.global_step = tf.train.get_or_create_global_step()


def train():

    global train_file
    global ckpt_dir

    num_train_epochs = 3
    warmup_proportion = 0.1
    learning_rate = 5e-5
    batch_size = 36
    length = 2000#173108
    num_schema = len(schema2id)

    train_dataset = input_fn_builder(
        input_file=train_file,
        sequence_length = sequence_length,
        num_labels = num_schema,
        batch_size=batch_size,
        is_training=True,
        epochs=num_train_epochs)

    num_train_steps = int(
        length / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()

    model = BertClassifier(bert_config, iterator.get_next(),  num_schema, init_checkpoint,is_training=True,
                    learning_rate=learning_rate,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps)
    gpu_options = tf.GPUOptions(allow_growth=True)
    print("OOOOOOOOOOOKKKKKKKKKKKKK")
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        g_list = tf.global_variables()

        saver = tf.train.Saver(g_list, max_to_keep=3)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore = {}".format(ckpt.model_checkpoint_path))
        train_handle = sess.run(train_iterator.string_handle())
        epoch_step = int(length / batch_size)
        current_epoch = 1
        # 用于tensorboard查看使用
        total_loss_summary = tf.summary.scalar("total_loss", model.total_loss)
        train_summary_op = tf.summary.merge([total_loss_summary])
        train_summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
        while True:
            for _ in tqdm.tqdm(range(epoch_step), desc="train epoch {},".format(current_epoch)):
                _,loss,global_step,train_summary= sess.run([model.train_op,model.total_loss,model.global_step,train_summary_op],feed_dict={handle:train_handle})
                #print(global_step,loss)
                if global_step % epoch_step == 0:
                    current_epoch += 1
                if global_step % 10 == 0:
                    #用于tensorboard查看使用
                    train_summary_writer.add_summary(train_summary, global_step=global_step)
                if global_step % 5 == 0:
                    checkpoint_prefix = os.path.join(ckpt_dir, 'classifier')
                    saver.save(sess, checkpoint_prefix, global_step=global_step)

def freeze_graph():

    features = {
        'input_ids': tf.placeholder(shape=[1, sequence_length], dtype=tf.int32, name='input_ids'),
        'input_mask': tf.placeholder(shape=[1, sequence_length], dtype=tf.int32, name='input_mask'),
        'segment_ids': tf.placeholder(shape=[1, sequence_length], dtype=tf.int32, name='segment_ids'),
    }
    model = BertClassifier(bert_config, features, len(id2schema), init_checkpoint,is_training=False)
    if not os.path.isdir('./data/graph'):
        os.makedirs('./data/graph')
    output_graph = os.path.join('./data/graph', "classifier.pb")
    output_node_names = "logits"  # 原模型输出操作节点的名字

    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path
    print(input_checkpoint)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()
    with tf.Session().as_default() as sess:

        g_list = tf.global_variables()
        saver = tf.train.Saver(g_list, max_to_keep=3)
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点







if __name__ == '__main__':
    import os, re
    import json
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from optparse import OptionParser


    parser = OptionParser()
    parser.add_option("--type", type="string", dest="type", default='train', help="类型")
    (options, args) = parser.parse_args()

    if options.type == 'corpus':
        write_tfrecord(corpus_file,train_file,sequence_length)
    if options.type == 'train':
        train()
    if options.type == 'freeze':
        freeze_graph()



    pass
