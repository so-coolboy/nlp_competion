# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Chile
# @Email   : realchilewang@foxmail.com
# @Software: PyCharm

from config import Config
import codecs
import numpy as np
import re
import os
import json
from snippts import split_text


def get_train_path_list(data_dir):
    type_map = {"txt": 1, "ann": 2}
    file_list = []
    for i in os.listdir(data_dir):
        file_list.append(i)
    file_list.sort(key=lambda x:
                (int(re.findall(r"""(\d+).(.*)""", x, re.DOTALL)[0][0]),
                 type_map[re.findall(r"""(\d+).(.*)""", x, re.DOTALL)[0][1]],
                 )
                )
    file_list = list(map(lambda x: data_dir  + x, file_list))
    text_list,text_label = [],[]
    for index,item in  enumerate(file_list):
        if index%2 == 0:
            text_list.append(item)
        else:
            text_label.append(item)
    data_file = []
    for index,item in  enumerate(text_list):

        data_file.append( (text_list[index],text_label[index]))

    return data_file


def get_test_path_list(data_dir):

    file_list = []
    for i in os.listdir(data_dir):
        file_list.append(i)
    file_list.sort(key=lambda x:
                int(re.findall(r"""(\d+).(.*)""", x, re.DOTALL)[0][0])
                )
    file_list = list(map(lambda x: data_dir  + x, file_list))

    return file_list

def remove_perfect_html_label(text):
    # 去掉完整的html标签
    html_labels = list(set(re.findall(r"""(<[^>]+>)""", text, re.DOTALL)))
    if html_labels:
        for html_label in html_labels:
            text = text.replace(html_label, "§" * len(html_label))
    text = text.replace(" ", "§").replace("\u3000", "§")
    return text


def remove_imperfect_html_label(remove_pattern, text):
    final_list = re.findall(remove_pattern[0], text, re.DOTALL)
    html_labels = [remove_pattern[1] + i + remove_pattern[2] for i in final_list]

    for html_label in html_labels:
        text = text.replace(html_label, "§" * len(html_label))
    return text


def remove_imperfect_html_labels(text):
    html_replaces = [(r"\?(.*?)\?/td>", "?", "?/td>"), (r"\?(.*?)\?br/>", "?", "?br/>")]
    for html_replace in html_replaces:
        text = remove_imperfect_html_label(html_replace, text)
    return text


def test_data_clean(text):
    html_replaces = [r'<[^>]+>', r"\?(.*?)\?/td>", r"\?(.*?)\?br/>", r"""&.*?；"""]
    correct_lists = [(" ", ""), ("\u3000", ""),
                     ("&ldquo;", '"'), ("&rdquo;", '"'),
                     ("&#40;", '('), ("&#41;", ')'),
                     ("&alpha;", 'α'), ("&beta;", 'β'),
                     ("&mdash;", ""), ("&times;", ""),
                     ("&quot;", "")
                     ]
    # text = text.replace(" ", "").replace("\u3000", "")
    for correct_char in correct_lists:
        text = text.replace(correct_char[0], correct_char[1])
    for i in html_replaces:
        text = re.sub(i, "", text)
    # 去除一些乱码的unicode
    # 中文的范围为 [\u4e00-\u9fa5]
    text = re.sub(r'[\ue000-\uffff]', "", text)

    # 清洗html标签
    return text


# 一些html的转义字符
# #“ &ldquo;“”
# ” &rdquo;
# &mdash; 破折号
# ( &#40; — 小括号左边部分Left parenthesis
# ) &#41; — 小括号右边部分Right parenthesis
# α &alpha;
# × &#215; &times; 乘号Multiply sign
# β &beta;
def correct_escape_char(text):
    correct_lists = [("&ldquo;", "§" * (len("&ldquo;") - 1) + '"'), ("&rdquo;", '"' + "§" * (len("&rdquo;") - 1)),
                     ("&#40;", "§" * (len("&#40;") - 1) + '('), ("&#41;", ')' + "§" * (len("&#41;") - 1)),
                     ("&alpha;", "§" * (len("&alpha;") - 1) + 'α'), ("&beta;", "§" * (len("&beta;") - 1) + 'β'),
                     ("&mdash;", "§" * len("&mdash;")), ("&times;", "§" * len("&times;")),
                     ]
    for correct_char in correct_lists:
        text = text.replace(correct_char[0], correct_char[1])
    return text


def read_single_train_data(file_path):
    # 读取训练集的文本数据
    with codecs.open(file_path[0], 'r', encoding="utf-8") as f:
        text = ""
        for line in f:
            text += line
    # 清洗html标签
    text = remove_perfect_html_label(text)
    text = remove_imperfect_html_labels(text)
    # 清洗html转义字符
    text = correct_escape_char(text)
    # 去除一些乱码的unicode
    # 中文的范围为 [\u4e00-\u9fa5]
    text = re.sub(r'[\ue000-\uffff]', "§", text)

    # print(text)

    # 读取标签数据
    label_dict = dict()
    with codecs.open(file_path[1], 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            items = line[1].split(" ")
            assert int(items[2]) - int(items[1]) == len(line[-1])
            if items[0] not in label_dict:
                label_dict[items[0]] = list()
                label_dict[items[0]].append({"start": int(items[1]), "end": int(items[2]), "content": line[-1]})
            else:
                label_dict[items[0]].append({"start": int(items[1]), "end": int(items[2]), "content": line[-1]})

    text_label = ["O"] * len(text)
    for label_type, items in label_dict.items():
        for item in items:
            assert text[item["start"]:item["end"]] == item["content"]

            for make_label in range(item["start"], item["end"]):
                text_label[make_label] = "I-" + label_type
            text_label[item["start"]] = "B-" + label_type

    # 230    [312,291,266,277]
    # 240    [255,312,291,266,277]
    # 250    [261,312,263,254,291,266,277]
    sub_texts, starts = split_text(text, 400, split_pat=None, greedy=False)
    # sub_texts, starts = split_text(text, 200, split_pat=None, greedy=True)

    texts, labels = [], []
    for sub_index, sub_text in enumerate(sub_texts):
        temp_text = list(sub_text)
        temp_label = text_label[starts[sub_index]:starts[sub_index] + len(sub_text)]
        # if len(sub_text) > 253:
        # print(len(sub_text))
        # assert len(sub_text)<= 510
        assert len(temp_text) == len(temp_label)
        texts.append(temp_text)
        labels.append(temp_label)
    return texts, labels

if __name__ == '__main__':

    # 设置随机种子
    np.random.seed(2020)
    config = Config()

    train_file = get_train_path_list(config.source_train_dir)
    test_file = get_test_path_list(config.source_test_dir)

    with codecs.open(config.processed_data + "train.txt", "w", encoding="utf-8") as f:
        num_list = []
        for i in train_file:
            texts, labels = read_single_train_data(i)
            for text_index, text in enumerate(texts):
                if len(text) > 512:
                    num_list.append(len(text))
                for index, item in enumerate(text):
                    if text[index] == "§" and labels[text_index][index] != "O":
                        print("数据处理异常！！！")
                    if text[index] != "§":
                        f.write('{0} {1}\n'.format(text[index], labels[text_index][index]))
                f.write('\n')
        print(num_list)

    with codecs.open(config.processed_data + "dev.txt", "w", encoding="utf-8") as f:
        train_list_index = list(range(len(train_file)))
        np.random.shuffle(train_list_index)
        train_list_index = train_list_index[:int(len(train_list_index) * 0.2)]

        for i in train_list_index:
            texts, labels = read_single_train_data(train_file[i])
            for text_index, text in enumerate(texts):
                for index, item in enumerate(text):
                    if text[index] == "§" and labels[text_index][index] != "O":
                        print("数据处理异常！！！")
                    if text[index] != "§":
                        f.write('{0} {1}\n'.format(text[index], labels[text_index][index]))
                f.write('\n')

    with codecs.open(config.processed_data + "test.txt", "w", encoding="utf-8") as fw:
        record_json = []
        num_list = []
        for file_name in test_file:
            test_name = file_name.split("/")[-1]
            with codecs.open(file_name, 'r', encoding="utf-8") as fr:
                text = ""
                for line in fr:
                    text += line
            text = test_data_clean(text)
            sub_texts, starts = split_text(text, 400, split_pat=None, greedy=False)
            lens = [len(sub_text) for sub_text in sub_texts]

            temp_num_list = [i for i in lens if i > 512]
            num_list.extend(temp_num_list)

            record_json.append({"file_name": test_name, "starts": starts, "lens": lens})
            for sub_text in sub_texts:
                sub_text = list(sub_text)
                for c1 in sub_text:
                    fw.write('{0} {1}\n'.format(c1, 'O'))
                fw.write('\n')
        print(num_list)

    with codecs.open(config.processed_data + "record_json.json", "w", encoding="utf-8") as fw:
        json.dump(record_json, fw, ensure_ascii=False)