# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Chile
# @Email   : realchilewang@foxmail.com
# @Software: PyCharm

from config import Config
from predict import extract_entity
import pandas as pd


def get_clean_csv(file_path):

    text_list = []

    all_tokens_list = []
    all_label_token_list = []

    with open(file_path, encoding='utf8') as fr:
        label_token_list = []
        tokens_list = []
        for line in fr:
            line = line.strip()
            if line:
                # print(line)
                token, label_token = line.split(' ')
                tokens_list.append(token)
                label_token_list.append(label_token)
            else:
                text_list.append("".join(tokens_list))
                all_tokens_list.append(tokens_list)
                all_label_token_list.append(label_token_list)


                label_token_list = []
                tokens_list = []

    # for index, token_list in enumerate(all_tokens_list[:146]):
    #     label_token_list = all_label_token_list[index]
    #     with open(data_dir + 'test.txt', 'a', encoding='utf-8') as fw:
    #         for i, token in enumerate(token_list):
    #             label_token = label_token_list[i]
    #             fw.write(token + ' ' + label_token + '\n')
    #         fw.write('\n')

    label_list = extract_entity(all_label_token_list, all_tokens_list)
    return text_list, label_list

if __name__ == '__main__':
    config = Config()
    data_dir = config.processed_data
    text_l, label_l = get_clean_csv(data_dir + 'source_train.txt')
    dev_text_l, dev_label_l = get_clean_csv(data_dir + 'dev.txt')

    text_l.extend(dev_text_l)
    label_l.extend(dev_label_l)
    final_label_list = []
    for l_l in label_l:
        temp_list = []
        for key in l_l:
            temp_list.extend(l_l[key])
        final_label_list.append(";".join(temp_list))

    data_dict = {'text': text_l, 'unknownEntities': final_label_list}
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(data_dir + 'new_train_df.csv')