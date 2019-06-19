#!/usr/bin/python3
#--coding:utf-8--
#@file       : extractor.py
#@author     : chenwanyuan
#@date       :2019/06/10/
#@description:
#@other      :
from tensorflow.python.framework import graph_util
import os
from collections import defaultdict
from bert import modeling
import json
import tqdm
import tensorflow as tf
from bert import tokenization
from bert import optimization


bert_config_file = './data/chinese_L-12_H-768_A-12/bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = './data/chinese_L-12_H-768_A-12/bert_model.ckpt'
schema_file = './data/all_50_schemas'
bert_tokenizer = tokenization.FullTokenizer(
        vocab_file="./data/chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=True)


label2id = {'0': 0, 'B-S': 1, "I-S": 2, 'B-O': 3, 'I-O': 4}
id2label = {}
for label, id in label2id.items():
    id2label[id] = label
sequence_length = 256
ckpt_dir = './extractor'
train_file = 'data/train_extractor.tfrecord'
corpus_file = './data/train_data.json'


def find_all(text,patten):
    idx = 0
    idxs = []
    while idx < len(text):
        i = text[idx:].find(patten)
        if i < 0:
            break
        else:
            idxs.append(idx + i)
            idx = idx + i + len(patten)
    return idxs


def write_tfrecord(file,tfrecord_file,max_sequence_length = 256):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    num = 0
    with open(file,encoding='utf8') as fp:
        for q_id,line in enumerate(fp):
            if num > 2000:break
            data = json.loads(line.strip())


            spo_list = data['spo_list']

            text = data['text']
            lower_text = text.lower()

            spo_map = defaultdict(list)
            for spo in spo_list:
                s_t = spo['subject_type']
                o_t = spo['object_type']
                p = spo['predicate']
                s = spo['subject']
                o = spo['object']
                spo_name = '{}-{}-{}'.format(s_t, p, o_t)
                spo_map[spo_name].append(spo)


            for  i, (spo_name, spo_list) in enumerate(spo_map.items()):

                tags = {}

                for spo in spo_list:
                    s_t = spo['subject_type']
                    o_t = spo['object_type']
                    p = spo['predicate']
                    s = spo['subject']
                    o = spo['object']
                    s_idxs = find_all(lower_text,s.lower())
                    o_idxs = find_all(lower_text,o.lower())

                    if len(s_idxs) > 0 and len(o_idxs) >0:
                        for idx in s_idxs:
                            if idx in tags:
                                if len(tags[idx][0]) < len(s):
                                    tags[idx] = s,'S'
                            else:
                                tags[idx] = s, 'S'
                        for idx in o_idxs:
                            if idx in tags:
                                if len(tags[idx][0]) < len(o):
                                    tags[idx] = o, 'O'
                            else:
                                tags[idx] = o, 'O'

                if len(tags) ==0:
                    print(data)
                    continue

                content_tokens = []
                content_label_ids = []

                pre_idx = 0
                pre_name = ''
                for i,(idx,(name ,t)) in enumerate(sorted(tags.items(), key=lambda d:d[0])):
                    if i ==0:
                        pre_text = lower_text[:idx]
                    else:
                        pre_text = lower_text[pre_idx + len(pre_name):idx]

                    pre_tokens = bert_tokenizer.tokenize(pre_text)

                    content_tokens.extend(pre_tokens)
                    content_label_ids.extend([0]*len(pre_tokens))

                    name_text = lower_text[idx:idx+len(name)]
                    name_tokens = bert_tokenizer.tokenize(name_text)
                    name_label_ids = [label2id['B-'+t]] + [label2id['I-'+t]]*(len(name_tokens)-1)

                    content_tokens.extend(name_tokens)
                    content_label_ids.extend(name_label_ids)

                    pre_idx = idx
                    pre_name = name

                if lower_text[-len(name):] != name:
                    last_text = lower_text[pre_idx + len(pre_name):]
                    last_tokens = bert_tokenizer.tokenize(last_text)
                    content_tokens.extend(last_tokens)
                    content_label_ids.extend([0] * len(last_tokens))

                spo_name_tokens = bert_tokenizer.tokenize(spo_name)
                tokens = ['[CLS]'] + spo_name_tokens + ['[SEP]'] + content_tokens + ['[SEP]']
                segment_ids = [0] * (len(spo_name_tokens) + 2) + [1] * (len(content_tokens) + 1)
                input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
                token_label_ids = [0] * (len(spo_name_tokens) +2) + content_label_ids + [0]
                input_mask = [1] * len(input_ids)

                if len(input_ids) < max_sequence_length:
                    input_ids = input_ids + [0] * (max_sequence_length - len(input_ids))
                    token_label_ids = token_label_ids + [0] * (max_sequence_length - len(token_label_ids))
                    input_mask = input_mask + [0] * (max_sequence_length - len(input_mask))
                    segment_ids = segment_ids + [1] * (max_sequence_length - len(segment_ids))
                else:
                    input_ids = input_ids[:max_sequence_length]
                    token_label_ids = token_label_ids[:max_sequence_length]
                    input_mask = input_mask[:max_sequence_length]
                    segment_ids = segment_ids[:max_sequence_length]
                    # print(len(input_ids),len(token_label_ids),len(input_mask),len(segment_ids))
                length = min(len(tokens), max_sequence_length)
                features = {}
                features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
                features["token_label_ids"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(token_label_ids)))
                features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_mask)))
                features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(segment_ids)))
                features["sequence_length"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[length]))
                features["q_id"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[q_id]))
                num += 1
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
    writer.close()
    print(num)




def input_fn_builder(input_file,max_sequence_length,batch_size=32,is_training=True,epochs=1):

    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_sequence_length], tf.int64),
        "token_label_ids": tf.FixedLenFeature([max_sequence_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_sequence_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_sequence_length], tf.int64),
        "sequence_length": tf.FixedLenFeature([], tf.int64),
        "q_id": tf.FixedLenFeature([], tf.int64),
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







def crf(input_ids,labels,sequence_lengths,num_labels):
    #with tf.variable_scope('proj'):
    W = tf.get_variable(name='W',
                        shape=[input_ids.get_shape().as_list()[-1],num_labels ],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32)

    b = tf.get_variable(name='b',
                        shape=[num_labels],
                        initializer=tf.zeros_initializer(),
                        dtype=tf.float32)

    input_ids_reshape = tf.reshape(input_ids, [-1, input_ids.get_shape().as_list()[-1]])
    logits = tf.reshape(tf.matmul(input_ids_reshape, W) + b, [-1, input_ids.get_shape().as_list()[1], num_labels], name='logits')



    #with tf.variable_scope("crf_loss"):
    trans = tf.get_variable(
        "transitions",
        shape=[num_labels, num_labels],
        initializer=tf.contrib.layers.xavier_initializer())
    decode_tags,best_score = tf.contrib.crf.crf_decode(logits, trans, sequence_lengths)
    tf.identity(decode_tags,'decode_tags')
    if labels is None:
        return None, logits,trans,decode_tags,best_score
    else:
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=labels,
            transition_params=trans,
            sequence_lengths=sequence_lengths)
        return tf.reduce_mean(-log_likelihood), logits,trans,decode_tags,best_score


class BertExtractor():
    def __init__(self,bert_config,features,num_labels,init_checkpoint,is_training=False,learning_rate=None,num_train_steps=None,num_warmup_steps=None):
        self.input_ids = features["input_ids"]
        self.input_mask = features["input_mask"]
        self.segment_ids = features["segment_ids"]
        #self.token_tag_ids = features["token_tag_ids"]

        self.token_label_ids = features.get("token_label_ids",None)
        self.sequence_lengths = features.get("sequence_length")
        print(features)
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        token_label_output_layer = model.get_sequence_output()

        self.total_loss,_,_,self.token_label_predictions,_ = crf(token_label_output_layer,self.token_label_ids,self.sequence_lengths,
                                                                 num_labels)
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



    num_train_epochs = 5
    warmup_proportion = 0.1
    learning_rate = 5e-5
    batch_size = 36
    length = 2000#173108
    train_dataset = input_fn_builder(
        input_file=train_file,
        max_sequence_length = sequence_length,
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

    model = BertExtractor(bert_config, iterator.get_next(), len(id2label), init_checkpoint,is_training=True,
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
                    train_summary_writer.add_summary(train_summary, global_step=global_step)
                if global_step % 5 == 0:
                    checkpoint_prefix = os.path.join(ckpt_dir, 'bert_crf')
                    saver.save(sess, checkpoint_prefix, global_step=global_step)
def freeze_graph():
    features = {
        'input_ids': tf.placeholder(shape=[1, sequence_length], dtype=tf.int32, name='input_ids'),
        'input_mask': tf.placeholder(shape=[1, sequence_length], dtype=tf.int32, name='input_mask'),
        'segment_ids': tf.placeholder(shape=[1, sequence_length], dtype=tf.int32, name='segment_ids'),
        'sequence_length': tf.placeholder(shape=[1], dtype=tf.int32, name='sequence_length'),
    }
    model = BertExtractor(bert_config, features, len(id2label), init_checkpoint, is_training=False)
    if not os.path.isdir('../data/graph'):
        os.makedirs('../data/graph')
    output_graph = os.path.join('../data/graph', "extractor.pb")
    output_node_names = "decode_tags"  # 原模型输出操作节点的名字


    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()
    with tf.Session().as_default() as sess:
        g_list = tf.global_variables()
        saver = tf.train.Saver(g_list, max_to_keep=3)
        print(ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore = {}".format(ckpt.model_checkpoint_path))

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点




    pass
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #train()
    #exit()

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--type", type="string", dest="type", default='train', help="类型")
    (options, args) = parser.parse_args()

    if options.type == 'corpus':
        write_tfrecord(corpus_file, train_file, sequence_length)
    if options.type == 'train':
        train()
    if options.type == 'freeze':
        freeze_graph()
