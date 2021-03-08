# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Xiachuankun
# @Email   : 1364707232@qq.com
# @Software: Vs Code

import os
import time
from tqdm import tqdm
import torch
from config import Config
import random
from sklearn.metrics import classification_report
import logging
from NEZHA.model_nezha import BertConfig
from NEZHA import nezha_utils
from model import BertForTokenClassification
from utils import DataIterator
import numpy as np
from transformers import BertTokenizer, RobertaConfig, AlbertConfig
from optimization import BertAdam
from utils import get_labels
import pickle

gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().processed_data
print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Batch Size: ', Config().batch_size)
print('Use original bert', Config().use_origin_bert)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# 固定每次结果
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)  # 我的理解是：确保每次实验结果一致，不设置实验的情况下准确率这些指标会有波动，因为是随机


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list))
    return target_ts.to(device)

# train 代码没有设置早停等技巧
def train(train_iter, test_iter, config):
    """"""
    # Prepare model
    # Prepare model
    # reload weights from restore_file if specified  如果指定就加载已经训练的权重
    if config.pretrainning_model == 'nezha':   #哪吒模型
        Bert_config = BertConfig.from_json_file(config.bert_config_file)
        model = BertForTokenClassification(config=Bert_config, params=config)
        nezha_utils.torch_init_model(model, config.bert_file)
    elif config.pretrainning_model == 'albert':
        Bert_config = AlbertConfig.from_pretrained(config.model_path)
        model = BertForTokenClassification.from_pretrained(config.model_path, config=Bert_config)
    else:
        Bert_config = RobertaConfig.from_pretrained(config.bert_config_file, output_hidden_states=True)
        model = BertForTokenClassification.from_pretrained(config=Bert_config, params=config,
                                                           pretrained_model_name_or_path=config.model_path)
   
    Bert_config.output_hidden_states = True  # 获取每一层的输出

    model.to(device)

    """多卡训练"""
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # optimizer
    # Prepare optimizer
    # fine-tuning
    # 取模型权重
    param_optimizer = list(model.named_parameters())
    # pretrain model param       预训练的参数
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n or 'electra' in n]  # nezha的命名为bert
    # middle model param         中等参数
    param_middle = [(n, p) for n, p in param_optimizer if
                    not any([s in n for s in ('bert', 'crf', 'electra', 'albert')]) or 'dym_weight' in n]
    # crf param
    # 不进行衰减的权重
    no_decay = ['bias', 'LayerNorm', 'dym_weight', 'layer_norm']
    # 将权重分组
    optimizer_grouped_parameters = [
        # pretrain model param  预训练的参数
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.embed_learning_rate
         },
        # 不衰减
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.embed_learning_rate
         },
        # middle model     中等参数
        # 衰减
        {'params': [p for n, p in param_middle if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.learning_rate
         },
        # 不衰减
        {'params': [p for n, p in param_middle if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.learning_rate
         },
    ]
    num_train_optimization_steps = train_iter.num_records // config.gradient_accumulation_steps * config.train_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=config.warmup_proportion, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps)
    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", config.batch_size)
    logger.info("  Num epochs = %d", config.train_epoch)
    logger.info("  Learning rate = %f", config.learning_rate)

    cum_step = 0
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(
        os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to {}\n".format(out_dir))

    draw_step_list = []
    draw_loss_list = []
    for i in range(config.train_epoch):
        model.train()
        for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list in tqdm(train_iter):
            # 转成张量
            loss = model(input_ids=list2ts2device(input_ids_list), token_type_ids=list2ts2device(segment_ids_list),
                         attention_mask=list2ts2device(input_mask_list), labels=list2ts2device(label_ids_list))
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # 梯度累加
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            if cum_step % 10 == 0:
                draw_step_list.append(cum_step)
                draw_loss_list.append(loss)
                if cum_step % 100 == 0:
                    format_str = 'step {}, loss {:.4f} lr {:.5f}'
                    print(
                        format_str.format(
                            cum_step, loss, config.learning_rate)
                    )

            loss.backward()  # 反向传播，得到正常的grad
            if (cum_step + 1) % config.gradient_accumulation_steps == 0:
                # performs updates using calculated gradients
                optimizer.step()
                model.zero_grad()
            cum_step += 1
        p, r, f1 = set_test(model, test_iter)
        # lr_scheduler学习率递减 step

        print('dev set : step_{},precision_{}, recall_{}, F1_{}'.format(cum_step, p, r, f1))

        # 保存模型
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            os.path.join(out_dir, 'model_{:.4f}_{:.4f}_{:.4f}_{}.bin'.format(p, r, f1, str(cum_step))))
        torch.save(model_to_save, output_model_file)

    with open(Config().processed_data + 'step_loss_data.pickle', 'wb') as mf:
        draw_dict = {'step': draw_step_list, 'loss': draw_loss_list}
        pickle.dump(draw_dict, mf)


def set_test(model, test_iter):
    if not test_iter.is_test:
        test_iter.is_test = True
    labels = get_labels()
    idx2tag = dict(zip(range(len(labels)), labels))
    model.eval()
    with torch.no_grad():
        true_tags, pred_tags = [], []
        for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list in tqdm(test_iter):
            input_ids = list2ts2device(input_ids_list)
            input_mask = list2ts2device(input_mask_list)
            segment_ids = list2ts2device(segment_ids_list)
            batch_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            # 恢复标签真实长度
            real_batch_tags = []
            for i in range(config.batch_size):
                real_len = int(input_mask[i].sum())
                real_batch_tags.append(label_ids_list[i][:real_len])

            # List[int]
            pred_tags.extend([idx2tag.get(idx) for indices in batch_output for idx in indices])
            true_tags.extend([idx2tag.get(idx) for indices in real_batch_tags for idx in indices])
            assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'
        # logging loss, f1 and report

        target_names = set(config.tags) - {"[PAD]", "[CLS]", "[SEP]", "O"}
        evaluation_dict = classification_report(true_tags, pred_tags, digits=4, output_dict=True)
        precision = 0
        recall = 0
        f1 = 0
        for key in evaluation_dict.keys():
            if key in target_names:
                precision += evaluation_dict[key]['precision']
                recall += evaluation_dict[key]['recall']
                f1 += evaluation_dict[key]['f1-score']
        f1 = f1 / len(target_names)
        precision = precision / len(target_names)
        recall = recall / len(target_names)

        print('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))
        return precision, recall, f1


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
    train(train_iter, dev_iter, config=config)
