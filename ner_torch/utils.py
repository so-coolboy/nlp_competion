# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Chile
# @Email   : realchilewang@foxmail.com
# @Software: PyCharm

import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from config import Config


def load_data(data_file_name):
    with open(data_file_name,'r',encoding='utf-8') as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            if len(contends) == 0:    #ner数据在一个与另一个数据之间使用一个“\n”隔开。所以当它为0时，代表这个数据已经结束了。
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
    return lines



# 把每一行数据都包装成这样的一个类对象
def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label
def create_example(lines):
    examples = []
    for (index, line) in enumerate(lines):
        guid = "%s" % index
        label = line[0]
        text = line[1]
        examples.append(InputExample(guid=guid, text=text, label=label))
    return examples

def get_labels():
    return Config().tags


class DataIterator(object):
    """
    数据迭代器
    """

    def __init__(self, batch_size, data_file, tokenizer, pretrainning_model=False, seq_length=100, is_test=False):
        # 数据文件位置
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.pretrainning_model = pretrainning_model
        self.seq_length = seq_length

        # 数据的个数
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        self.label_map = {}
        for index, label in enumerate(get_labels()):   #标签编码一下
            self.label_map[label] = index
        print("标签个数：", len(get_labels()))
        print("样本个数：", self.num_records)

    def convert_single_example(self, example_idx):
        '''
        根据传入的idx，将这条数据转换成需要的格式
        '''
        text_list = self.data[example_idx].text.split(" ")            # 数据格式由['', '', '']变成['选',
        label_list = self.data[example_idx].label.split(" ")    # 标签                            '基',
        tokens = text_list               #                                                        '金']
        labels = label_list

        # seq_length=128 则最多有126个字符
        if len(tokens) >= self.seq_length - 1:
            tokens = tokens[:(self.seq_length - 2)]
            labels = labels[:(self.seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        
        # 首先添加CLS的标记，分别在tokens， segment和labels中添加，要对应
        ntokens.append('[CLS]')
        segment_ids.append(0)
        label_ids.append(self.label_map['[CLS]'])

        for index, token in enumerate(tokens):
            try:
                ntokens.append(self.tokenizer.tokenize(token.lower())[0])  # 全部转换成小写, 方便BERT词典
            except:
                ntokens.append('[UNK]')    # 能用分词就分词，不能就标记不认识
            segment_ids.append(0)          # 对于ner任务，全部填零就可以了
            label_ids.append(self.label_map[labels[index]])

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[SEP]"])

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)  #Bert的字典编码，把中文的字编码成数字
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.seq_length:   #padding
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(self.label_map["[PAD]"])
            ntokens.append("*NULL*")
            tokens.append("*NULL*")
        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        assert len(label_ids) == self.seq_length
        assert len(tokens) == self.seq_length
        return input_ids, input_mask, segment_ids, label_ids, tokens

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件，所有的数据都迭代过了
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        label_ids_list = []
        tokens_list = []

        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, label_ids, tokens = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            tokens_list.append(tokens)

            if self.pretrainning_model:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break

        while len(input_ids_list) < self.batch_size: #如果还少数据，就把第一条数据拿出来重复
            input_ids_list.append(input_ids_list[0])
            input_mask_list.append(input_mask_list[0])
            segment_ids_list.append(segment_ids_list[0])
            label_ids_list.append(label_ids_list[0])
            tokens_list.append(tokens_list[0])

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    train_iter = DataIterator(config.batch_size,
                              data_file=config.processed_data + 'new_train.txt',
                              pretrainning_model=config.pretrainning_model,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'dev.txt',
                            pretrainning_model=config.pretrainning_model,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list in tqdm(dev_iter):
        print(input_ids_list[-1])
        print(label_ids_list[-1])
        break

