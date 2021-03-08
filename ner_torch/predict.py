# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Xiachuankun
# @Email   : 1364707232@qq.com
# @Software: Vs Code

import torch
from tqdm import tqdm
from config import Config
from transformers import BertTokenizer
from utils import DataIterator
from utils import get_labels
import logging
import os
import numpy as np
import json
from sklearn.metrics import classification_report
gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().processed_data
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list))
    return target_ts.to(device)


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def extract_entity(pred_tags, tokens_list):
    """
    将Bio数据转换为实体
    :param pred_tags:
    :param params:
    :return:
    """
    pred_result=[]
    for idx, line in enumerate(pred_tags):
        # get BIO-tag
        entities = get_entities(line)
        sample_dict={}
        for entity in entities:
            label_type = entity[0]
            if label_type=='[CLS]' or label_type=='[SEP]':
                continue
            start_ind = entity[1]
            end_ind = entity[2]
            en = tokens_list[idx][start_ind:end_ind + 1]
            if label_type in sample_dict.keys():
                sample_dict[label_type].append(''.join(en))
            else:
                sample_dict[label_type]=[''.join(en)]
        pred_result.append(sample_dict)
    return pred_result


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        # >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    # print(seq)
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def set_test(test_iter, model_file):
    model = torch.load(model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("***** Running Prediction *****")
    logger.info("  Predict Path = %s", model_file)
    model.eval()
    labels = get_labels()
    idx2tag = dict(zip(range(len(labels)), labels))
    pred_answer = []
    true_tags, pred_tags = [], []
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list in tqdm(
            test_iter):
        input_ids = list2ts2device(input_ids_list)
        input_mask = list2ts2device(input_mask_list)
        segment_ids = list2ts2device(segment_ids_list)
        batch_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # 恢复标签真实长度
        real_batch_tags = []
        for i in range(config.batch_size):
            real_len = int(input_mask[i].sum())
            real_batch_tags.append(label_ids_list[i][:real_len])

        pred_tags.extend([idx2tag.get(idx) for indices in batch_output for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in real_batch_tags for idx in indices])
        assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'
        pred = [[idx2tag.get(idx) for idx in indices] for indices in batch_output]
        answer_batch = extract_entity(pred, tokens_list)
        pred_answer.extend(answer_batch)
    print()
    target_names = set(config.tags) - {"[PAD]", "[CLS]", "[SEP]", "O"}
    print(classification_report(true_tags, pred_tags, digits=4))
    evaluation_dict = classification_report(true_tags, pred_tags, digits=4, output_dict=True)
    precision = 0
    recall = 0
    f1 = 0
    new_ev_dict = dict()
    for key in evaluation_dict.keys():
        if key in target_names:
            precision += evaluation_dict[key]['precision']
            recall += evaluation_dict[key]['recall']
            f1 += evaluation_dict[key]['f1-score']

            if key.split('-')[-1] in new_ev_dict.keys():
                new_ev_dict[key.split('-')[-1]]['precision'].append(evaluation_dict[key]['precision'])
                new_ev_dict[key.split('-')[-1]]['recall'].append(evaluation_dict[key]['recall'])
                new_ev_dict[key.split('-')[-1]]['f1-score'].append(evaluation_dict[key]['f1-score'])
            else:
                new_ev_dict[key.split('-')[-1]] = dict()
                new_ev_dict[key.split('-')[-1]]['precision'] = [evaluation_dict[key]['precision']]
                new_ev_dict[key.split('-')[-1]]['recall'] = [evaluation_dict[key]['recall']]
                new_ev_dict[key.split('-')[-1]]['f1-score'] = [evaluation_dict[key]['precision']]

    final_ev_dict = dict()
    for key in new_ev_dict.keys():
        ev = new_ev_dict[key]
        final_ev_dict[key] = dict()
        for e_key in ev.keys():
            final_ev_dict[key][e_key] = round(sum(ev[e_key]) / len(ev[e_key]), 4)
    # print(final_ev_dict)
    for key in final_ev_dict.keys():
        ev_p = final_ev_dict[key]['precision']
        ev_r = final_ev_dict[key]['recall']
        ev_f1 = final_ev_dict[key]['f1-score']
        print(key, ev_p, ev_r, ev_f1)

    f1 = f1 / len(target_names)
    precision = precision / len(target_names)
    recall = recall / len(target_names)

    print('{:.4f} {:.4f} {:.4f}'.format(precision, recall, f1))
    with open(result_data_dir + 'result.json', 'w', encoding='utf—8') as f:
        json.dump(pred_answer, f, ensure_ascii=False)


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size,
                            config.processed_data + 'dev.txt',
                            pretrainning_model=config.pretrainning_model,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    set_test(dev_iter, config.checkpoint_path)
